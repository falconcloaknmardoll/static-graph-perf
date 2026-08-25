"""
dataset_oracle.py — Phase 8d: TPUGraphDataset + per-config oracle rank-bucket feature.

NEW FILE by design (mirrors dataset_mxu.py) — dataset.py/train.py stay untouched.
Only imported by train_8d.py / phase8d_evaluate.py.

Per experiments/phase8d_preregistration.md §4 (frozen, before running):
  • For each graph FILE, sort all configs in THAT file by true config_runtime and
    assign K=5 quantile rank-buckets (fastest 20%=bucket 0, ..., slowest 20%=
    bucket 4). Buckets are computed per-file, i.e. within whichever split
    (train/valid — a graph belongs to exactly one split directory) that file
    already sits in — this IS the "train configs bucketed against train-split
    runtimes, val configs against val-split runtimes" requirement, since a
    file's own config_runtime array only ever contains that file's own split.
  • Unlike the per-node, config-invariant MXU features in dataset_mxu.py, the
    bucket is a PER-CONFIG signal, so it is appended as a one-hot 5-dim block
    scattered onto the same config-node rows as node_config_feat (Early-Join
    pattern: dataset.py's `cfg_pad[config_ids] = cfg_feat`), not to every node.
    158 -> 163.
  • This is an ORACLE, never a deployable feature — true runtime is not known
    at inference time in any real deployment. Every reference to this arm in
    the paper must say so explicitly (prereg §4).
"""
from __future__ import annotations

import numpy as np
import torch

from dataset import TPUGraphDataset

K_BUCKETS = 5


class OracleGraphDataset(TPUGraphDataset):
    """TPUGraphDataset with a per-config one-hot rank-bucket block appended to x."""

    @property
    def num_node_features(self) -> int:  # 158 + 5
        return self.TOTAL_FEAT_DIM + K_BUCKETS

    def _get_graph_cache(self, file_idx: int) -> dict:
        cache = super()._get_graph_cache(file_idx)
        if "oracle_bucket" not in cache:
            npz = self._get_handle(file_idx)
            runtime = np.asarray(npz["config_runtime"], dtype=np.float64)
            # Rank-based bucketing (not a literal quantile cut) so ties never
            # collapse configs into one bucket: sort ascending, split into K
            # contiguous rank chunks of (near-)equal size.
            order = np.argsort(runtime, kind="stable")
            n = len(runtime)
            bucket = np.empty(n, dtype=np.int64)
            bucket[order] = np.minimum((np.arange(n) * K_BUCKETS) // n, K_BUCKETS - 1)
            cache["oracle_bucket"] = bucket
        return cache

    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        file_idx, cfg_idx = self._index[idx]
        cache = self._get_graph_cache(file_idx)
        bucket = int(cache["oracle_bucket"][cfg_idx])
        N = data.x.shape[0]
        onehot = torch.zeros(K_BUCKETS, dtype=data.x.dtype)
        onehot[bucket] = 1.0
        onehot_pad = torch.zeros(N, K_BUCKETS, dtype=data.x.dtype)
        # Same bucket value at every config-node row (it doesn't vary by node,
        # only by config) — mirrors dataset.py's cfg_pad[config_ids] = cfg_feat.
        onehot_pad[torch.from_numpy(cache["config_ids"])] = onehot
        data.x = torch.cat([data.x, onehot_pad], dim=1)
        return data
