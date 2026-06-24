"""
test_dataset.py
---------------
Phase 0.4 – Data Pipeline Validation for TPUGraphs.

Covers six invariants that must hold before training:
  1. Feature Dimensions      – x is (N, 158), y is a scalar
  2. Early-Join Integrity    – config features are scattered to the right rows
  3. Edge Index Validity     – shape (2, E), all ids < N, no negatives
  4. Runtime Variance        – runtimes differ across configs of the same graph
  5. Performance / Memory    – 100 sequential items load quickly, no leak
  6. Single-Graph Batching   – PyG DataLoader produces correct batch vector & y

Run:
    pytest test_dataset.py -v
    pytest test_dataset.py -v -k test_early_join   # run one test
"""

from __future__ import annotations

import random
import time
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dataset import (
    TPUGraphDataset,
    MXU_OPS,
    MEM_OPS,
    MXU_WINDOW,
    MEM_WINDOW,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = (
    "data/predict-ai-model-runtime/npz_all/npz/layout/xla/random/train"
)

# ---------------------------------------------------------------------------
# Module-scoped fixture — dataset is opened once per pytest session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> TPUGraphDataset:
    """Instantiate the train split a single time for the whole test module."""
    return TPUGraphDataset(data_dir=DATA_DIR)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _indices_for_graph(ds: TPUGraphDataset, file_idx: int) -> List[int]:
    """Return every flat dataset index whose (file_idx, *) matches `file_idx`."""
    return [i for i, (f, _) in enumerate(ds._index) if f == file_idx]


def _find_graph_with_at_least(ds: TPUGraphDataset, n: int) -> int:
    """Return the first file_idx that has >= n configurations, or skip."""
    for file_idx in range(ds.num_files):
        if len(_indices_for_graph(ds, file_idx)) >= n:
            return file_idx
    pytest.skip(f"No graph with >= {n} configs found in train split.")


# ===========================================================================
# Test 1 – Feature Dimensions
# ===========================================================================

def test_feature_dimensions(dataset: TPUGraphDataset) -> None:
    """
    For 3 randomly sampled items:
      - data.x must have shape (N, 158)
      - data.y must be a 0-dimensional (scalar) float32 tensor
    """
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), k=3)

    for idx in indices:
        data = dataset[idx]
        N = data.num_nodes

        assert data.x.shape == (N, 158), (
            f"idx={idx}: expected x.shape=({N}, 158), got {tuple(data.x.shape)}"
        )
        assert data.x.dtype == torch.float32, (
            f"idx={idx}: x.dtype should be float32, got {data.x.dtype}"
        )
        assert data.y.dim() == 0, (
            f"idx={idx}: y must be a 0-dim scalar, got {data.y.dim()}-dim tensor"
        )
        assert data.y.dtype == torch.float32, (
            f"idx={idx}: y.dtype should be float32, got {data.y.dtype}"
        )


# ===========================================================================
# Test 2 – Early-Join Integrity
# ===========================================================================

def test_early_join_integrity(dataset: TPUGraphDataset) -> None:
    """
    Two configs from the same graph must satisfy:

      a) Base features (cols 0:140) are identical at every node — they come
         from the shared node_feat array and must never be modified.

      b) Non-config nodes have all-zero config slice (cols 140:158) in both
         configs, so their full 158-dim rows are byte-for-byte identical.

      c) Config-node rows (rows in node_config_ids) may differ in cols 140:158
         across the two configs — that is the *point* of Early-Join — but
         their base slice (cols 0:140) must still match.
    """
    file_idx = _find_graph_with_at_least(dataset, n=2)
    idx_a, idx_b = _indices_for_graph(dataset, file_idx)[:2]

    data_a = dataset[idx_a]
    data_b = dataset[idx_b]
    N = data_a.num_nodes
    assert data_b.num_nodes == N, "Sanity: both configs must belong to the same graph"

    # ── (a) base features identical everywhere ─────────────────────────────
    assert torch.equal(data_a.x[:, :140], data_b.x[:, :140]), (
        "Base node features (cols 0:140) diverge across configs of the same graph — "
        "Early-Join is corrupting the base array."
    )

    # ── build config-node mask from the raw npz ────────────────────────────
    npz = dataset._get_handle(file_idx)
    config_ids = torch.from_numpy(np.array(npz["node_config_ids"], dtype=np.int64))
    config_mask = torch.zeros(N, dtype=torch.bool)
    config_mask[config_ids] = True  # True  → config-affected node

    # ── (b) non-config rows must be fully identical ────────────────────────
    assert torch.equal(data_a.x[~config_mask], data_b.x[~config_mask]), (
        "Non-config nodes should be fully identical across configs (base feat + zero pad)."
    )

    # ── (c) config-node base slice must still match ────────────────────────
    assert torch.equal(data_a.x[config_mask, :140], data_b.x[config_mask, :140]), (
        "Base features (cols 0:140) at config nodes must be identical across configs."
    )

    # ── verify Early-Join actually wrote something non-zero for at least one
    #    config (guards against a silent all-zeros scatter bug) ─────────────
    cfg_slice_a = data_a.x[config_mask, 140:]  # (K, 18)
    cfg_slice_b = data_b.x[config_mask, 140:]  # (K, 18)
    assert cfg_slice_a.abs().sum() > 0 or cfg_slice_b.abs().sum() > 0, (
        "Config slice (cols 140:158) is all-zeros for both configs — "
        "Early-Join scatter appears to have done nothing."
    )


# ===========================================================================
# Test 3 – Edge Index Validity
# ===========================================================================

def test_edge_index_validity(dataset: TPUGraphDataset) -> None:
    """
    For 5 randomly sampled items:
      - edge_index.shape must be (2, E)
      - All node ids must be in [0, N-1] (no out-of-bounds references)
      - dtype must be int64
    """
    rng = random.Random(0)
    indices = rng.sample(range(len(dataset)), k=5)

    for idx in indices:
        data = dataset[idx]
        N = data.num_nodes
        ei = data.edge_index

        assert ei.dtype == torch.int64, (
            f"idx={idx}: edge_index.dtype must be int64, got {ei.dtype}"
        )
        assert ei.dim() == 2, (
            f"idx={idx}: edge_index must be 2-D, got {ei.dim()}-D"
        )
        assert ei.shape[0] == 2, (
            f"idx={idx}: edge_index.shape[0] must be 2 (PyG convention), "
            f"got {ei.shape[0]}.  Array may not have been transposed."
        )

        min_id = int(ei.min())
        max_id = int(ei.max())
        assert min_id >= 0, (
            f"idx={idx}: edge_index contains negative node id {min_id}"
        )
        assert max_id < N, (
            f"idx={idx}: edge_index contains node id {max_id} >= N={N} "
            "(out-of-bounds reference)"
        )


# ===========================================================================
# Test 4 – Runtime Variance
# ===========================================================================

def test_runtime_variance(dataset: TPUGraphDataset) -> None:
    """
    5 different configurations of the same graph must not all share the same
    target runtime.  Constant runtimes would indicate a data-loading bug and
    would make any ranking loss (ListMLE) trivial/degenerate.
    """
    file_idx = _find_graph_with_at_least(dataset, n=5)
    idx_list = _indices_for_graph(dataset, file_idx)[:5]

    runtimes = torch.stack([dataset[i].y for i in idx_list])

    assert not torch.all(runtimes == runtimes[0]), (
        f"All 5 configs of graph {file_idx} share the same runtime "
        f"{runtimes[0].item():.6e}.  Expected variance across configurations."
    )


# ===========================================================================
# Test 5 – Performance / Memory Sanity
# ===========================================================================

def test_performance_and_memory(dataset: TPUGraphDataset) -> None:
    """
    Load 100 sequential items and assert:
      1. No exception is raised.
      2. Wall-clock time < 30 s (reasonable upper bound on a laptop with mmap).
      3. [Optional, requires psutil] RSS growth < 500 MB — guards against
         inadvertent full-array materialisation inside the loop.
    """
    n_items = 100
    assert len(dataset) >= n_items, (
        f"Dataset has only {len(dataset)} items; need >= {n_items} for this test."
    )

    # Capture baseline memory if psutil is available
    proc = None
    mem_before_mb = 0.0
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        mem_before_mb = proc.memory_info().rss / 1024 ** 2
    except ImportError:
        pass  # psutil not installed — skip memory assertion

    start = time.perf_counter()
    for i in range(n_items):
        data = dataset[i]
        # Light sanity check so the loop body isn't trivially elided
        assert data.x.shape[1] == 158, f"Item {i}: unexpected feature dim"
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, (
        f"Loading {n_items} items took {elapsed:.2f} s — expected < 30 s.  "
        "Check for accidental full-array materialisation."
    )

    if proc is not None:
        mem_after_mb = proc.memory_info().rss / 1024 ** 2
        growth_mb = mem_after_mb - mem_before_mb
        assert growth_mb < 500, (
            f"RSS grew by {growth_mb:.1f} MB while loading {n_items} items — "
            "possible memory leak (mmap arrays being copied to RAM)."
        )


# ===========================================================================
# Test 6 – Single-Graph Batching (ListMLE prerequisite)
# ===========================================================================

def test_single_graph_batching(dataset: TPUGraphDataset) -> None:
    """
    Verify the PyG DataLoader correctly collates 4 configs from the same graph.

    ListMLE requires a batch that contains exactly one ranked list (= all
    configs of one graph).  This test confirms the collated batch satisfies:

      - batch.y.shape == (4,)                  one runtime per config
      - batch.batch.min() == 0
      - batch.batch.max() == 3                  indices 0–3
      - batch.batch.shape[0] == total node count
      - each batch id appears exactly num_nodes times
    """
    file_idx = _find_graph_with_at_least(dataset, n=4)
    idx_list = _indices_for_graph(dataset, file_idx)[:4]

    subset = Subset(dataset, idx_list)
    loader = DataLoader(subset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    # ── y vector ──────────────────────────────────────────────────────────
    assert batch.y.shape == (4,), (
        f"batch.y.shape should be (4,), got {tuple(batch.y.shape)}.  "
        "DataLoader may be stacking scalars incorrectly."
    )

    # ── batch vector range ────────────────────────────────────────────────
    assert int(batch.batch.min()) == 0, (
        f"batch.batch.min() should be 0, got {batch.batch.min().item()}"
    )
    assert int(batch.batch.max()) == 3, (
        f"batch.batch.max() should be 3, got {batch.batch.max().item()}.  "
        "Expected exactly 4 graphs in the batch."
    )

    # ── total node count ──────────────────────────────────────────────────
    expected_total = sum(dataset[i].num_nodes for i in idx_list)
    assert batch.batch.shape[0] == expected_total, (
        f"batch.batch has {batch.batch.shape[0]} entries but expected "
        f"{expected_total} (sum of num_nodes across 4 configs)."
    )

    # ── per-graph node count matches exactly ──────────────────────────────
    for local_idx, flat_idx in enumerate(idx_list):
        expected_count = dataset[flat_idx].num_nodes
        actual_count = int((batch.batch == local_idx).sum())
        assert actual_count == expected_count, (
            f"Config {local_idx} (flat idx {flat_idx}): batch vector assigns "
            f"{actual_count} nodes but num_nodes={expected_count}."
        )


# ===========================================================================
# Test 7 – Phase 3: Hardware-Aware Structural Edges
# ===========================================================================

def test_structural_edges(dataset: TPUGraphDataset) -> None:
    """
    Validate the Granite-style structural-edge augmentation on 5 graphs:

      a) Alignment   – edge_attr has shape (E', 3) and E' matches edge_index's
                       second dimension (one attribute row per edge).
      b) One-hot     – every edge_attr row sums to exactly 1.0 (valid one-hot,
                       no edge left untyped or double-typed).
      c) Bounded     – the augmentation never explodes into an O(N²) clique:
                          E' ≤ E_orig + 2·W_mxu·N_mxu + 2·W_mem·N_mem
                       where N_mxu / N_mem are the counts of MXU / MEM opcode
                       nodes and E_orig is the raw data-flow edge count.
    """
    rng = random.Random(7)
    file_indices = rng.sample(range(dataset.num_files), k=min(5, dataset.num_files))

    for file_idx in file_indices:
        flat_idx = _indices_for_graph(dataset, file_idx)[0]
        data = dataset[flat_idx]

        ei = data.edge_index            # (2, E')
        ea = data.edge_attr             # (E', 3)
        e_prime = ei.shape[1]

        # ── (a) alignment ──────────────────────────────────────────────────
        assert ea.dim() == 2 and ea.shape[1] == 3, (
            f"file {file_idx}: edge_attr must be (E', 3), got {tuple(ea.shape)}"
        )
        assert ea.shape[0] == e_prime, (
            f"file {file_idx}: edge_attr has {ea.shape[0]} rows but edge_index "
            f"has {e_prime} edges — one attribute row per edge is required."
        )
        assert ea.dtype == torch.float32, (
            f"file {file_idx}: edge_attr.dtype should be float32, got {ea.dtype}"
        )

        # ── (b) valid one-hot: every row sums to exactly 1.0 ───────────────
        row_sums = ea.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(e_prime), atol=0.0), (
            f"file {file_idx}: {(row_sums != 1.0).sum().item()} edge_attr rows "
            "do not sum to exactly 1.0 — one-hot edge typing is invalid."
        )
        # entries must themselves be 0/1, not e.g. [0.5, 0.5, 0.0]
        assert torch.all((ea == 0.0) | (ea == 1.0)), (
            f"file {file_idx}: edge_attr contains non-binary entries — not one-hot."
        )

        # ── (c) edge count is bounded (no O(N²) clique) ────────────────────
        npz = dataset._get_handle(file_idx)
        e_orig = int(np.array(npz["edge_index"]).shape[0])        # raw (E, 2)
        opcode = np.array(npz["node_opcode"], dtype=np.uint8)
        n_mxu = int(np.isin(opcode, list(MXU_OPS)).sum())
        n_mem = int(np.isin(opcode, list(MEM_OPS)).sum())

        upper_bound = e_orig + 2 * MXU_WINDOW * n_mxu + 2 * MEM_WINDOW * n_mem
        assert e_prime <= upper_bound, (
            f"file {file_idx}: E'={e_prime} exceeds the contention bound "
            f"{upper_bound} (E_orig={e_orig}, N_mxu={n_mxu}, N_mem={n_mem}). "
            "Edge construction is not local — possible O(N²) explosion."
        )
        # structural edges must actually be additive, never dropping data-flow
        assert e_prime >= e_orig, (
            f"file {file_idx}: E'={e_prime} < E_orig={e_orig} — augmentation "
            "dropped data-flow edges."
        )
