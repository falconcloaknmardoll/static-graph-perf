"""
dataset_mxu.py — Phase 8c: TPUGraphDataset + appended MXU-alignment features.

NEW FILE by design: dataset.py/train.py must stay byte-identical while capped
runs from other phases execute (each job re-imports them). This subclass is
imported ONLY by train_8c.py / phase8c evaluation.

Per the pre-registration (experiments/phase8c_preregistration.md §5):
  • MXU columns are appended to the existing 158-dim Early-Join input
    (158 → 158+k). Non-conv nodes get zeros.
  • Raw MXU values get the same sign·log1p normalization as other features.
  • Arms (exact column lists — recorded as prereg amendment A1):
      channel : c_out_mxu_util, c_out_pad_waste, k_mxu_util, k_pad_waste   (k=4)
      spatial : kernel_area, gemm_m, n_mxu_passes                          (k=3)
      full    : all 16 Phase-8b features                                   (k=16)
  • NaN contraction features (conv nodes whose kernel operand could not be
    resolved) → 0, counted per graph the first time each graph is loaded.

The MXU block is computed ONCE per graph (raw arrays re-read from the mmap
handle — the base cache stores log1p'd node_feat, but compute_mxu_features
needs RAW integer-valued sizes) and cached alongside the base graph cache, so
it invalidates on graph change exactly like the rest of the cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from dataset import TPUGraphDataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from mxu_features import compute_mxu_features, FEATURE_NAMES  # noqa: E402

ARM_COLUMNS = {
    "baseline": [],
    "channel": ["c_out_mxu_util", "c_out_pad_waste", "k_mxu_util", "k_pad_waste"],
    "spatial": ["kernel_area", "gemm_m", "n_mxu_passes"],
    "full": list(FEATURE_NAMES),
    # ── Specificity control (prereg amendment A2, registered 2026-07-20) ──
    # Identical columns/dim to `channel` (k=4), but within each graph the
    # feature ROWS are permuted AMONG the MXU-resolved (nonzero) nodes with a
    # deterministic per-graph RNG (seed = file_idx). This preserves value
    # distribution, conv-node placement, dimensionality, and config-invariance
    # while DESTROYING the node↔value alignment (the actual MXU information).
    # channel > scramble ⇒ real co-design signal; channel ≈ scramble ⇒ the 8c
    # effect is capacity/regularization from extra conv-node columns, not info.
    "scramble": ["c_out_mxu_util", "c_out_pad_waste", "k_mxu_util", "k_pad_waste"],
}

# Arms whose per-graph block is a row-scramble of their column values.
_SCRAMBLE_ARMS = {"scramble"}


class MXUGraphDataset(TPUGraphDataset):
    """TPUGraphDataset with a per-node MXU feature block appended to x."""

    def __init__(self, data_dir: str, mxu_arm: str = "full") -> None:
        if mxu_arm not in ARM_COLUMNS:
            raise ValueError(f"unknown arm {mxu_arm!r}; choose from {list(ARM_COLUMNS)}")
        self.mxu_arm = mxu_arm
        self._arm_cols = [FEATURE_NAMES.index(c) for c in ARM_COLUMNS[mxu_arm]]
        self._scramble = mxu_arm in _SCRAMBLE_ARMS
        super().__init__(data_dir=data_dir)

    # ── feature dimensionality ────────────────────────────────────────────
    @property
    def num_node_features(self) -> int:  # 158 + k
        return self.TOTAL_FEAT_DIM + len(self._arm_cols)

    # ── per-graph MXU block, cached with the base graph cache ─────────────
    def _get_graph_cache(self, file_idx: int) -> dict:
        cache = super()._get_graph_cache(file_idx)
        if self._arm_cols and "mxu_block" not in cache:
            npz = self._get_handle(file_idx)
            raw_feat = np.asarray(npz["node_feat"], dtype=np.float64)   # RAW sizes
            opcode = np.asarray(npz["node_opcode"])
            edge_index = np.asarray(npz["edge_index"])                  # (E,2) [consumer, operand]
            feats, _ = compute_mxu_features(raw_feat, opcode, edge_index)
            block = feats[:, self._arm_cols].astype(np.float32)
            n_nan = int(np.isnan(block).sum())
            if n_nan:
                block = np.nan_to_num(block, nan=0.0)   # unresolved kernels → 0 (prereg A1)
            # sign·log1p — same normalization as the base features.
            block = np.sign(block) * np.log1p(np.abs(block))
            if self._scramble:
                # Permute feature rows AMONG the MXU-resolved (any-nonzero) nodes
                # only, so conv-node placement + value distribution are preserved
                # and only the node↔value alignment (the information) is destroyed.
                # Deterministic per graph (seed=file_idx) → config-invariant and
                # identical across training seeds, exactly like `channel`'s data.
                nz = np.flatnonzero(np.any(block != 0.0, axis=1))
                if nz.size > 1:
                    perm = np.random.default_rng(file_idx).permutation(nz.size)
                    block[nz] = block[nz][perm]
            cache["mxu_block"] = torch.from_numpy(block)                # (N, k)
        return cache

    # ── append block to x ──────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        if self._arm_cols:
            file_idx, _ = self._index[idx]
            block = self._get_graph_cache(file_idx)["mxu_block"]
            data.x = torch.cat([data.x, block], dim=1)                  # (N, 158+k)
        return data
