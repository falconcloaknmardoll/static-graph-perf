"""
model.py
--------
Phase 1 Baseline: GraphSAGE + Global Mean Pool for TPUGraphs runtime prediction.

Architecture
~~~~~~~~~~~~
  Input (N, 158)  ← 140-dim base features + 18-dim config features (Early-Join)
      ↓  × 3
  SAGEConv → BatchNorm1d → ReLU → Dropout(0.1)          node-level, (N, 128)
      ↓
  global_mean_pool                                         graph-level, (K, 128)
      ↓
  Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, 1)  scalar score per config

Known limitations (Phase 1 — intentional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  • 3-layer receptive field vs ~80-hop graph diameter (from analyze_topology.py):
    nodes > 3 hops apart have zero Jacobian. The model is blind to most of the
    graph. This is the primary motivation for DiffPool in Phase 2.

  • Global mean pool dilutes the config-node signal: for AlexNet (N=372, K=26
    config nodes), only 26/372 = 7% of the pooled embedding carries
    configuration-specific information. The remaining 93% is identical across
    all 29,144 configurations of that graph.

Both limitations are kept deliberately to reproduce Google's published baseline
before measuring the delta from hierarchical pooling in Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class BaselineGraphSAGE(nn.Module):
    """GraphSAGE encoder + global mean readout for single-scalar scoring.

    Each forward pass receives a PyG-collated batch containing K configurations
    of the same computation graph (enforced by SingleGraphBatchSampler in
    train.py).  The model produces one score per configuration; higher scores
    indicate a predicted faster (lower-latency) execution.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.  Default 158 = 140 (base) + 18 (config,
        Early-Join scattered from node_config_feat).
    hidden_channels : int
        Width of all GNN layers and the first MLP layer.  Default 128.
    dropout : float
        Dropout probability after each GNN activation and inside the MLP.
        Default 0.1.
    """

    def __init__(
        self,
        in_channels: int = 158,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── GNN Encoder: 3 × SAGEConv ─────────────────────────────────────
        # SAGEConv computes:  h_v' = W1·h_v + W2·mean{h_u : u ∈ N(v)}
        # BatchNorm1d operates over all nodes across the full batch
        # (total_nodes = K × N), which is large enough for stable statistics.
        self.convs = nn.ModuleList([
            SAGEConv(in_channels,      hidden_channels),   # layer 1
            SAGEConv(hidden_channels,  hidden_channels),   # layer 2
            SAGEConv(hidden_channels,  hidden_channels),   # layer 3
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.BatchNorm1d(hidden_channels),
        ])
        self.dropout = nn.Dropout(p=dropout)

        # ── MLP Head: 2 linear layers → scalar ────────────────────────────
        # Operates on pooled graph-level embeddings (shape: K × hidden).
        # No BatchNorm here: at this stage the "batch size" is configs_per_batch
        # (16–32), which is too small for reliable BN running statistics.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,           # (total_nodes, in_channels)
        edge_index: torch.Tensor,  # (2, total_edges)
        batch: torch.Tensor,       # (total_nodes,)  — PyG batch vector
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (total_nodes, 158)
            Stacked node features for all K configurations in the batch.
            total_nodes = K × N (same graph repeated K times with different
            config features).
        edge_index : Tensor, shape (2, total_edges)
            Stacked and offset edge indices (PyG DataLoader handles offset).
        batch : Tensor, shape (total_nodes,)
            Assigns each node to its configuration index in [0, K-1].

        Returns
        -------
        scores : Tensor, shape (K,)
            One predicted ranking score per configuration.
            Higher score → model predicts faster (lower-latency) execution.
        """
        # ── 3 rounds of message passing ────────────────────────────────────
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # aggregate from neighbours
            x = bn(x)                 # normalise across all nodes in batch
            x = F.relu(x)
            x = self.dropout(x)

        # ── Graph-level readout ────────────────────────────────────────────
        # global_mean_pool: (total_nodes, H) → (K, H)
        # Each row is the mean node embedding for one configuration.
        graph_emb = global_mean_pool(x, batch)  # (K, hidden_channels)

        # ── Scalar score per configuration ─────────────────────────────────
        scores = self.mlp(graph_emb).squeeze(-1)  # (K,)
        return scores
