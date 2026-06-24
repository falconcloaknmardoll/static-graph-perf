"""
dataset.py
----------
Lazy PyTorch Dataset for the TPUGraphs layout:xla:random split.

Memory Strategy
~~~~~~~~~~~~~~~
Each .npz file contains O(10^4) configurations over O(10^3–10^4) nodes.
Materialising all Early-Join tensors up-front would require ~100s of GB.
Instead, every .npz is opened with mmap_mode='r' so arrays live on disk;
only the slices needed for a single __getitem__ call are paged into RAM.

Early-Join (per __getitem__)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For configuration i of a graph with N nodes:
  1.  base   = node_feat              → (N, 140)
  2.  cfg    = node_config_feat[i]    → (K, 18),  K = len(node_config_ids)
  3.  Scatter cfg into zero-pad       → (N, 18)   using node_config_ids
  4.  x      = concat([base, pad])    → (N, 158)

edge_index
~~~~~~~~~~
.npz stores edges as (E, 2); PyG convention is (2, E).  Transposed at load.
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_npz(path: str) -> np.lib.npyio.NpzFile:
    """Open an npz with memory-mapping so large arrays stay on disk."""
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Phase 3 — Hardware-aware structural edges (Granite-style)
# ---------------------------------------------------------------------------
# Data-flow edges model only producer→consumer dependencies. TPU latency is
# also driven by *resource contention*: ops hitting the same hardware unit
# (MXU systolic array, or HBM bandwidth) compete when scheduled close together.
# We inject two extra relation types as a sliding-window "contention chain"
# over same-resource ops in schedule order (node-id order ≈ HLO instruction
# order), bounded to O(W·N) edges so we never build an O(N²) clique.
#
# Window sizes are asymmetric by microarchitectural design: on TPU v3 the HBM
# can sustain far higher concurrency than the single systolic MXU, so the
# memory window is widened to capture true bandwidth saturation rather than
# just a couple of parallel loads.
#
# Fusion is deliberately excluded from MXU_OPS: it is a meta-op whose
# contents are opaque, so it cannot be honestly assigned to the MXU or HBM.
#
# !! Opcode-enum shift !!  The integers in node_opcode do NOT follow the
# standard alphabetical XLA HLO enum. TPUGraphs uses a CUSTOM index assignment
# defined in reference/tpu_graphs/process_data/xla/hlo_opcode.h (the V(name,id)
# X-macro lists). The ids below were extracted from that header by
# scripts/extract_opcode_map.py — DO NOT replace them with XLA source enums.
# Anchors: convolution=26, dot=34 (NOT 30/39). einsum is absent (XLA lowers
# einsum → dot).

MXU_OPS: frozenset = frozenset({
    26,   # convolution
    34,   # dot
})
MEM_OPS: frozenset = frozenset({
    13,   # broadcast
    22,   # concatenate
    27,   # copy
    35,   # dynamic-slice
    36,   # dynamic-update-slice
    42,   # gather
    62,   # pad
    75,   # reshape
    76,   # reverse
    82,   # scatter
    92,   # slice
    98,   # transpose
})

MXU_WINDOW: int = 5     # systolic MXU: low concurrency → short contention chain
MEM_WINDOW: int = 15    # HBM: high concurrency → wide window for saturation

# edge_attr one-hot lanes: [data_flow, mxu_contention, mem_contention]
_DATA = (1.0, 0.0, 0.0)
_MXU  = (0.0, 1.0, 0.0)
_MEM  = (0.0, 0.0, 1.0)


def _contention_edges(node_ids: np.ndarray, window: int) -> np.ndarray:
    """Bidirectional sliding-window chain over an *ordered* list of node ids.

    Connects each op to the next ``window`` same-resource ops (both directions),
    modeling that sequentially-scheduled ops queue for the same hardware unit.
    Returns (2, E_c) int64; (2, 0) if fewer than 2 nodes.
    """
    m = node_ids.shape[0]
    if m < 2:
        return np.empty((2, 0), dtype=np.int64)
    src, dst = [], []
    for off in range(1, min(window, m - 1) + 1):
        a, b = node_ids[:-off], node_ids[off:]
        src += [a, b]; dst += [b, a]            # forward + reverse (mutual)
    return np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)


def add_structural_edges(
    edge_index: np.ndarray,    # (2, E) int64 — data-flow edges (already transposed)
    node_opcode: np.ndarray,   # (N,) uint8
    mxu_window: int = MXU_WINDOW,
    mem_window: int = MEM_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment data-flow edges with MXU- and MEM-contention edges.

    Returns
    -------
    aug_edge_index : (2, E') int64   — data_flow ++ mxu ++ mem
    edge_attr      : (E', 3) float32 — one-hot [data_flow, mxu, mem]
    """
    mxu_nodes = np.where(np.isin(node_opcode, list(MXU_OPS)))[0]  # ascending → schedule order
    mem_nodes = np.where(np.isin(node_opcode, list(MEM_OPS)))[0]

    mxu_ei = _contention_edges(mxu_nodes, mxu_window)
    mem_ei = _contention_edges(mem_nodes, mem_window)

    aug_edge_index = np.concatenate([edge_index, mxu_ei, mem_ei], axis=1)
    edge_attr = np.concatenate([
        np.tile(_DATA, (edge_index.shape[1], 1)),
        np.tile(_MXU,  (mxu_ei.shape[1],   1)),
        np.tile(_MEM,  (mem_ei.shape[1],   1)),
    ], axis=0).astype(np.float32)

    return aug_edge_index, edge_attr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TPUGraphDataset(Dataset):
    """Flat (graph, config) dataset for one TPUGraphs split directory.

    Each item corresponds to a single compiler configuration applied to a
    single XLA computation graph.  The "Early-Join" fuses the per-node base
    features with the configuration features *before* message passing, as
    recommended by the TPUGraphs paper.

    Parameters
    ----------
    data_dir:
        Path to a directory of .npz files, e.g.
        ``data/predict-ai-model-runtime/npz_all/npz/layout/xla/random/train``
    """

    # Feature dimensionalities (fixed by the dataset spec)
    NODE_FEAT_DIM:   int = 140
    CONFIG_FEAT_DIM: int = 18
    TOTAL_FEAT_DIM:  int = NODE_FEAT_DIM + CONFIG_FEAT_DIM  # 158

    def __init__(self, data_dir: str) -> None:
        super().__init__()

        self.data_dir = os.path.abspath(data_dir)

        self._file_paths: List[str] = sorted(
            glob.glob(os.path.join(self.data_dir, "*.npz"))
        )
        if not self._file_paths:
            raise FileNotFoundError(
                f"No .npz files found under {self.data_dir!r}.\n"
                "Double-check your data_dir path."
            )

        # Build a flat index: list of (file_idx, config_idx) tuples.
        # Files are opened temporarily here just to read config counts, then
        # closed so the dataset remains picklable (required for num_workers>0
        # with spawn/forkserver multiprocessing contexts).
        self._index: List[Tuple[int, int]] = []
        for file_idx, path in enumerate(self._file_paths):
            npz = _open_npz(path)
            n_configs: int = npz["config_runtime"].shape[0]
            npz.close()
            for cfg_idx in range(n_configs):
                self._index.append((file_idx, cfg_idx))

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def _get_handle(self, file_idx: int) -> "np.lib.npyio.NpzFile":
        """Per-process cached mmap handle. Created on first access in each worker."""
        if not hasattr(self, "_local_handles"):
            self._local_handles: dict = {}
        if file_idx not in self._local_handles:
            self._local_handles[file_idx] = _open_npz(self._file_paths[file_idx])
        return self._local_handles[file_idx]

    def _get_graph_cache(self, file_idx: int) -> dict:
        """Per-worker cache of graph-level arrays shared across all configs of a graph.

        The SingleGraphBatchSampler guarantees all items in a batch share the
        same file_idx, so node_feat (~4 MB) and edge_index are loaded once per
        batch instead of once per config, giving a ~32x reduction in disk reads.
        Cache is keyed on file_idx and invalidated on graph change.
        """
        if not hasattr(self, "_graph_cache_key") or self._graph_cache_key != file_idx:
            npz = self._get_handle(file_idx)
            node_feat = np.array(npz["node_feat"], dtype=np.float32)
            # Pre-apply sign*log1p to the 140-dim base features here once per graph.
            # Without this, each of the 512 configs per graph would recompute
            # log1p over N×140 floats, causing a ~100x CPU bottleneck vs the
            # per-config log1p applied only to the 18 config dims in __getitem__.
            node_feat_log = np.sign(node_feat) * np.log1p(np.abs(node_feat))

            # Phase 3: augment data-flow edges with hardware-contention edges.
            # Structural edges depend only on node_opcode (config-independent),
            # so they are built once here per graph alongside node_feat — never
            # recomputed for each of the 512 configs in __getitem__.
            node_opcode = np.array(npz["node_opcode"], dtype=np.uint8)        # (N,)
            data_flow_ei = np.array(npz["edge_index"].T, dtype=np.int64)      # (2, E)
            aug_ei, edge_attr = add_structural_edges(data_flow_ei, node_opcode)

            self._graph_cache_key = file_idx
            self._graph_cache = {
                "node_feat":        node_feat_log,             # (N, 140) log-normalized
                "edge_index":       aug_ei,                    # (2, E') data_flow++mxu++mem
                "edge_attr":        edge_attr,                 # (E', 3) one-hot edge type
                "config_ids":       np.array(npz["node_config_ids"], dtype=np.int64),
                "node_config_feat": npz["node_config_feat"],   # mmap: slice per config
                "config_runtime":   npz["config_runtime"],     # mmap: index per config
            }
        return self._graph_cache

    def __getitem__(self, idx: int) -> Data:
        file_idx, cfg_idx = self._index[idx]
        cache = self._get_graph_cache(file_idx)

        # --- Base node features (cached — same for all configs of this graph) ---
        node_feat: np.ndarray = cache["node_feat"]   # (N, 140)
        N: int = node_feat.shape[0]

        # --- Configuration features for this one config (cheap mmap row read) ---
        cfg_feat: np.ndarray = np.array(
            cache["node_config_feat"][cfg_idx], dtype=np.float32
        )  # (K, 18)

        # --- Early-Join scatter ----------------------------------------
        cfg_pad = np.zeros((N, self.CONFIG_FEAT_DIM), dtype=np.float32)
        cfg_pad[cache["config_ids"]] = cfg_feat

        # --- Concatenate → (N, 158) — always a fresh allocation --------
        # node_feat is already sign*log1p normalized (done once per graph in cache).
        # Apply sign*log1p only to cfg_pad*100 here (N×18 ops per config, not N×158).
        cfg_scaled = cfg_pad * 100.0
        log_cfg: np.ndarray = np.sign(cfg_scaled) * np.log1p(np.abs(cfg_scaled))
        x: np.ndarray = np.concatenate([node_feat, log_cfg], axis=1)

        # --- Config-node mask (same for all configs of this graph) ----
        # Boolean mask identifying the K_c configurable nodes out of N.
        # Stored per-item so PyG's Batch stacks it alongside x/edge_index.
        config_mask = torch.zeros(N, dtype=torch.bool)
        config_mask[torch.from_numpy(cache["config_ids"])] = True

        # --- Target latency --------------------------------------------
        y: float = float(cache["config_runtime"][cfg_idx])

        return Data(
            x=torch.from_numpy(x),                                   # (N, 158)
            edge_index=torch.from_numpy(cache["edge_index"].copy()),  # (2, E')
            edge_attr=torch.from_numpy(cache["edge_attr"].copy()),    # (E', 3) one-hot
            y=torch.tensor(y, dtype=torch.float32),                   # scalar
            config_mask=config_mask,                                  # (N,) bool
            num_nodes=N,
        )

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def num_node_features(self) -> int:
        """Dimensionality of node feature vectors after Early-Join (158)."""
        return self.TOTAL_FEAT_DIM

    @property
    def num_files(self) -> int:
        """Number of unique computation graphs in this split."""
        return len(self._file_paths)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_files={self.num_files}, "
            f"num_configs={len(self)}, "
            f"node_feat_dim={self.num_node_features})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = os.path.join(
        "data",
        "predict-ai-model-runtime",
        "npz_all",
        "npz",
        "layout",
        "xla",
        "random",
        "train",
    )

    print(f"Instantiating TPUGraphDataset from:\n  {os.path.abspath(DATA_DIR)}\n")
    dataset = TPUGraphDataset(data_dir=DATA_DIR)
    print(dataset)
    print(f"Total (graph, config) pairs : {len(dataset):,}")

    sample = dataset[0]
    print(f"\n--- dataset[0] ---")
    print(f"  x.shape          : {tuple(sample.x.shape)}   (expected: (N, 158))")
    print(f"  edge_index.shape : {tuple(sample.edge_index.shape)}")
    print(f"  y (latency)      : {sample.y.item():.4f}")
    print(f"  num_nodes        : {sample.num_nodes}")

    assert sample.x.shape[1] == TPUGraphDataset.TOTAL_FEAT_DIM, (
        f"Feature dim mismatch: got {sample.x.shape[1]}, "
        f"expected {TPUGraphDataset.TOTAL_FEAT_DIM}"
    )
    print("\nAssertion passed: feature dimensionality is correct.")
