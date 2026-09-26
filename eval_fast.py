"""
eval_fast.py — drop-in faster equivalent of train.evaluate().

train.evaluate() materialises one PyG Data object per (graph, config): a fresh
(N, 158) feature array plus copies of the (2, E') edge_index and (E', 3)
edge_attr, then merges chunks of them with Batch.from_data_list on the CPU. On
nlp collections (~70k val configs per graph) that CPU work dominates; the GPU
sits near idle.

Here everything config-independent (log-normalised node features, augmented
edges, edge types, config-node ids) is moved to the GPU once per graph, and
each chunk is assembled on-device. Only the (B, K, 18) config rows cross from
the CPU, and they are log-transformed with the same numpy ops as
TPUGraphDataset.__getitem__.

Semantics kept identical to train.evaluate():
  - same chunking (eval_batch_size consecutive configs, same order),
  - same batched layout as Batch.from_data_list (node blocks concatenated,
    edge_index offset by b*N, batch vector = graph id),
  - same Kendall τ call, same NaN-dropping, same 0.0 fallback.
Verify with scripts/verify_eval_fast.py before relying on it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn

from dataset import TPUGraphDataset
from model_phase3 import HardwareAwareDiffPool


class _GraphOnDevice:
    """Config-independent tensors of one graph, with per-chunk-size layouts cached."""

    def __init__(self, ds: TPUGraphDataset, file_idx: int, device: torch.device):
        cache = ds._get_graph_cache(file_idx)
        self.N = cache["node_feat"].shape[0]
        self.node_feat = torch.from_numpy(cache["node_feat"]).to(device)           # (N, 140)
        self.edge_index = torch.from_numpy(cache["edge_index"]).to(device)         # (2, E')
        self.edge_attr = torch.from_numpy(cache["edge_attr"]).to(device)           # (E', 3)
        self.config_ids = torch.from_numpy(cache["config_ids"]).to(device)         # (K,)
        self.node_config_feat = cache["node_config_feat"]                          # (C, K, 18)
        self.device = device
        self._layouts: Dict[int, tuple] = {}

    def layout(self, B: int):
        """Edge index / edge attr / batch vector / config mask for a chunk of B configs."""
        if B not in self._layouts:
            N, dev = self.N, self.device
            offs = (torch.arange(B, device=dev) * N).repeat_interleave(self.edge_index.shape[1])
            ei = self.edge_index.repeat(1, B) + offs
            ea = self.edge_attr.repeat(B, 1)
            batch = torch.arange(B, device=dev).repeat_interleave(N)
            mask = torch.zeros(B * N, dtype=torch.bool, device=dev)
            rows = (self.config_ids.unsqueeze(0)
                    + (torch.arange(B, device=dev) * N).unsqueeze(1)).reshape(-1)
            mask[rows] = True
            self._layouts[B] = (ei, ea, batch, mask, rows)
        return self._layouts[B]

    def features(self, cfg_indices: List[int]) -> torch.Tensor:
        """(B*N, 158) Early-Join features for these configs, identical to __getitem__."""
        B, N = len(cfg_indices), self.N
        cfg = np.array(self.node_config_feat[cfg_indices], dtype=np.float32)       # (B, K, 18)
        cfg_scaled = cfg * 100.0
        log_cfg = np.sign(cfg_scaled) * np.log1p(np.abs(cfg_scaled))              # same ops as dataset
        _, _, _, _, rows = self.layout(B)
        cfg_part = torch.zeros(B * N, TPUGraphDataset.CONFIG_FEAT_DIM, device=self.device)
        cfg_part[rows] = torch.from_numpy(log_cfg.reshape(-1, log_cfg.shape[-1])).to(self.device)
        return torch.cat([self.node_feat.repeat(B, 1), cfg_part], dim=1)


@torch.no_grad()
def evaluate_fast(
    model: nn.Module,
    val_dataset: TPUGraphDataset,
    device: torch.device,
    eval_batch_size: int = 32,
    return_preds: bool = False,
):
    """Same contract as train.evaluate(); optionally also returns per-graph predictions."""
    model.eval()
    groups: Dict[int, List[int]] = defaultdict(list)
    for flat_idx, (file_idx, _) in enumerate(val_dataset._index):
        groups[file_idx].append(flat_idx)

    taus: List[float] = []
    preds_by_graph: Dict[int, np.ndarray] = {}
    edge_aware = isinstance(model, HardwareAwareDiffPool)

    for file_idx, flat_indices in groups.items():
        if len(flat_indices) < 2:
            continue
        cfg_indices = [val_dataset._index[i][1] for i in flat_indices]
        npz = val_dataset._get_handle(file_idx)
        true_runtimes = np.array(npz["config_runtime"], dtype=np.float64)[cfg_indices]

        g = _GraphOnDevice(val_dataset, file_idx, device)
        all_preds: List[torch.Tensor] = []
        for start in range(0, len(cfg_indices), eval_batch_size):
            chunk = cfg_indices[start:start + eval_batch_size]
            x = g.features(chunk)
            ei, ea, batch, mask, _ = g.layout(len(chunk))
            if edge_aware:
                out = model(x, ei, batch, mask, ea)
            else:
                out = model(x, ei, batch, mask)
            preds = out[0] if isinstance(out, tuple) else out
            all_preds.append(preds.cpu())
        del g

        preds_np = torch.cat(all_preds).numpy()
        if return_preds:
            preds_by_graph[file_idx] = preds_np
        tau, _ = sp_stats.kendalltau(preds_np, -true_runtimes)
        if not np.isnan(tau):
            taus.append(float(tau))

    mean_tau = float(np.mean(taus)) if taus else 0.0
    result = {"val_tau": mean_tau, "val_n_graphs": len(taus)}
    return (result, preds_by_graph) if return_preds else result
