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

        # Open every file once (mmap — negligible RAM).  Keep handles alive
        # for the lifetime of the dataset so __getitem__ never re-opens.
        self._npz_handles: List[np.lib.npyio.NpzFile] = [
            _open_npz(p) for p in self._file_paths
        ]

        # Build a flat index: list of (file_idx, config_idx) tuples.
        # Reading config_runtime.shape[0] is cheap — it's a tiny 1-D array.
        self._index: List[Tuple[int, int]] = []
        for file_idx, npz in enumerate(self._npz_handles):
            n_configs: int = npz["config_runtime"].shape[0]
            for cfg_idx in range(n_configs):
                self._index.append((file_idx, cfg_idx))

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Data:
        file_idx, cfg_idx = self._index[idx]
        npz = self._npz_handles[file_idx]

        # --- Base node features ----------------------------------------
        # Materialise a fresh copy; mmap keeps the source on disk.
        node_feat: np.ndarray = np.array(
            npz["node_feat"], dtype=np.float32
        )  # (N, 140)
        N: int = node_feat.shape[0]

        # --- Configuration features for this one config ----------------
        # Slicing a single row of a mmap array pages in only that row.
        config_ids: np.ndarray = np.array(
            npz["node_config_ids"], dtype=np.int64
        )  # (K,)
        cfg_feat: np.ndarray = np.array(
            npz["node_config_feat"][cfg_idx], dtype=np.float32
        )  # (K, 18)

        # --- Early-Join scatter ----------------------------------------
        # Allocate zeros for all N nodes, then scatter K config rows in.
        cfg_pad = np.zeros((N, self.CONFIG_FEAT_DIM), dtype=np.float32)
        cfg_pad[config_ids] = cfg_feat  # scatter: only K rows are non-zero

        # --- Concatenate → (N, 158) ------------------------------------
        x: np.ndarray = np.concatenate([node_feat, cfg_pad], axis=1)

        # --- Edge index ------------------------------------------------
        # .npz: (E, 2)  →  PyG: (2, E)
        edge_index: np.ndarray = np.array(
            npz["edge_index"].T, dtype=np.int64
        )  # (2, E)

        # --- Target latency --------------------------------------------
        y: float = float(npz["config_runtime"][cfg_idx])

        return Data(
            x=torch.from_numpy(x),                           # (N, 158)
            edge_index=torch.from_numpy(edge_index),         # (2, E)
            y=torch.tensor(y, dtype=torch.float32),          # scalar
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
