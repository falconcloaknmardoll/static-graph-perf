"""
model.py
--------
GraphSAGE baseline with residual connections for TPUGraphs runtime prediction.

Architecture
~~~~~~~~~~~~
  Input (N, 158)  ← 140-dim base features + 18-dim config features × 100 (Early-Join)
      ↓
  Linear(158→128)                                    input projection
      ↓  × 3
  SAGEConv(128→128) + BN(no running stats) + LeakyReLU + Dropout + residual
      ↓
  global_mean_pool  (all nodes)      (K, 128)
  global_mean_pool  (config nodes)   (K, 128)
  concat → (K, 256)
      ↓
  Linear(256→64) + ReLU + Dropout + Linear(64→1)   scalar per config

Key design notes
~~~~~~~~~~~~~~~~
  • BatchNorm with track_running_stats=False: BN always uses the *current batch*
    statistics (both in train and eval mode).  Default BN accumulates running
    mean/var during training and switches to those in eval — but the base node
    features span [-2, 26M], so running stats accumulated over 69 training graphs
    are often wrong for any individual val graph.  Setting track_running_stats=False
    makes BN behaviour identical in train and eval: normalise by whatever batch
    you have right now.  Requires eval batches of ≥ 2 items (eval_batch=32 in
    train.py satisfies this).

  • Residual connections: x_{l+1} = x_l + dropout(LeakyReLU(BN(SAGEConv(x_l)))).
    The additive skip keeps config features (100x-amplified at input) alive
    through all 3 GNN layers.  Residuals also stabilise gradients without needing
    the learning-rate surgery that's required in the no-norm variant.

  • LeakyReLU (negative_slope=0.01): prevents dead neurons that occur when
    large initial activations push units into negative territory and keep them
    stuck.

  • Dual pooling: all-node pool (context) + config-node-only pool (config signal).
    Without the config-only pool the ~7% of configurable nodes are diluted 14:1
    by non-configurable nodes in global_mean_pool, making configurations
    indistinguishable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class BaselineGraphSAGE(nn.Module):
    """GraphSAGE encoder with residual connections + dual pooling.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension (158 = 140 base + 18 config × 100).
    hidden_channels : int
        GNN and first MLP hidden width.  Default 128.
    dropout : float
        Dropout probability after each GNN activation and in the MLP.
    """

    def __init__(
        self,
        in_channels: int = 158,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Input projection ──────────────────────────────────────────────────
        # Brings 158-dim Early-Join features to hidden_channels so that all
        # subsequent residual additions are between equal-dimensional tensors.
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ── GNN Encoder: 3 × SAGEConv with BatchNorm (no running stats) ──────
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
        ])
        # track_running_stats=False: always use current-batch statistics.
        # Eliminates the train/eval BN mismatch caused by diverse graph scales.
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
        ])
        self.dropout = nn.Dropout(p=dropout)

        # ── MLP Head ──────────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,                # (total_nodes, in_channels)
        edge_index: torch.Tensor,       # (2, total_edges)
        batch: torch.Tensor,            # (total_nodes,)
        config_mask: torch.Tensor,      # (total_nodes,) bool
    ) -> torch.Tensor:
        """Returns shape (K,) — one score per configuration in the batch."""

        # ── Input projection ──────────────────────────────────────────────────
        x = F.leaky_relu(self.input_proj(x), negative_slope=0.01)

        # ── 3 GNN layers with residual connections ────────────────────────────
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.leaky_relu(h, negative_slope=0.01)
            h = self.dropout(h)
            x = x + h   # residual: config signal survives all 3 layers

        # ── Dual graph-level readout ──────────────────────────────────────────
        all_emb    = global_mean_pool(x, batch)                        # (K, H)
        config_emb = global_mean_pool(x[config_mask], batch[config_mask])  # (K, H)
        graph_emb  = torch.cat([all_emb, config_emb], dim=-1)         # (K, 2H)

        return self.mlp(graph_emb).squeeze(-1)                         # (K,)


class BaselineGraphSAGE_Large(BaselineGraphSAGE):
    """Parameter-matched wide baseline for the DiffPool ablation study.

    hidden_channels=172 → 236,157 params; HierarchicalGraphSAGE at h=128
    has 235,553 params (delta: +604, <0.3%).  Architecture is otherwise
    identical (3× SAGEConv, dual pooling, same MLP structure), so any τ
    gap between this model and DiffPool is attributable to hierarchy, not
    to parameter capacity.

    Use with --model wide-baseline in train.py.
    """

    HIDDEN: int = 172   # the unique hidden_channels that matches DiffPool param count

    def __init__(
        self,
        in_channels: int = 158,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=self.HIDDEN,
            dropout=dropout,
        )
