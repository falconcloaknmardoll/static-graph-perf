"""
model_phase2.py
---------------
Phase 2 — HierarchicalGraphSAGE with memory-safe DiffPool.

Architecture
~~~~~~~~~~~~
  Input (N, 158)
      → SAGEConv          [pre-pool GNN]
      → DiffPool 1        [K1 = min(⌈0.25·N⌉, K1_MAX) clusters]
      → SAGEConv          [post-pool-1 GNN]
      → DiffPool 2        [K2 = min(⌈0.25·K1⌉, K2_MAX) clusters]
      → SAGEConv          [post-pool-2 GNN]
      → Global Mean Pool
      → 2-layer MLP       [→ scalar score]

The OOM Trap — and how we escape it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standard DiffPool (Ying et al. 2018) computes  A_new = S^T A S  and
X_new = S^T Z  using *dense* N×N adjacency matrices (PyG's
dense_diff_pool / to_dense_batch).  For BERT-scale XLA graphs (N ≈ 40,000):

    dense float32 adjacency = 40,000² × 4 B = 6.4 GB  — instant OOM.

Our memory-safe approach keeps A in sparse COO (edge_index) format and
computes  A @ S  via a single scatter_add over the edge list:

    (A @ S)[i, k] = Σ_{j : (i,j) ∈ E}  S[j, k]

Cost: O(|E| × K) time, O(N × K) extra memory — no dense N×N ever built.
Then  A_pool = S^T (A @ S)  is a dense (B, K, K) batched matmul, tiny
because K ≤ K1_MAX = 128.

Cluster budget cap (K1_MAX / K2_MAX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
True 25% reduction on N = 40,000 would give K = 10,000.
The soft-assignment matrix S would then be  B × N × K = 8 × 40,000 × 10,000
× 4 B ≈ 12 GB — entirely impossible.

We fix K1_MAX = 128 and K2_MAX = 32.  For large graphs the cap activates
(coarser pooling than 25%); for small/medium graphs (N ≤ 512) true 25%
reduction applies automatically via  K = min(int(0.25 × N), K_MAX).

Memory at the cap (worst case B=8, N=40,000):
    S  : 8 × 40,000 × 128 × 4 B = 164 MB   ← the limiting tensor
    AS : same = 164 MB
    A_pool : 8 × 128 × 128 × 4 B = 0.5 MB  ← negligible

This fits within 16 GB VRAM alongside the forward activations and optimizer
state.  If still OOM on BERT-scale graphs, reduce --configs-per-batch to 4.

Auxiliary losses
~~~~~~~~~~~~~~~~
forward() returns (scores, aux_losses) where aux_losses contains:

  lp_loss      Link Prediction Loss   ||A − SS^T||_F^2 / N^2
               Encourages edges to connect nodes that share a cluster.
               Computed without any N×N materialization:
                   LP² = |E| − 2·tr(S^T A S) + ||S^T S||_F²
               (derivation in DiffPoolLayer._aux_losses)

  entropy_loss Entropy Regularization   H(S) = −Σ_i S_i log S_i / N
               Pushes the soft assignment toward one-hot (sharp clusters).

Both are sums over the two DiffPool layers.  Train with:
    loss = listmle_loss + λ_lp · lp_loss + λ_ent · entropy_loss
Typical starting values: λ_lp = 0.1, λ_ent = 0.01.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_scatter import scatter


K1_MAX: int = 128   # cluster budget after first DiffPool
K2_MAX: int = 32    # cluster budget after second DiffPool


# ─────────────────────────────────────────────────────────────────────────────
# DiffPool layer
# ─────────────────────────────────────────────────────────────────────────────

class DiffPoolLayer(nn.Module):
    """
    One memory-safe DiffPool layer.

    Two parallel GNNs per Ying et al. (2018):
      z_gnn  — new cluster embeddings:  Z = LeakyReLU(BN(SAGEConv(X, A)))
      s_gnn  — soft assignment matrix:  S = softmax(SAGEConv(X, A))

    A @ S is computed via scatter_add over the COO edge list (no dense N×N).

    Parameters
    ----------
    in_channels      : input feature dimension
    hidden_channels  : output feature dimension (Z dimension)
    k_max            : hard upper bound on number of clusters
    ratio            : target fraction of nodes to keep (0.25 = 25%)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        k_max: int,
        ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.k_max = k_max
        self.ratio = ratio

        # Z-GNN
        self.z_conv = SAGEConv(in_channels, hidden_channels)
        self.z_bn   = nn.BatchNorm1d(hidden_channels, track_running_stats=False)

        # S-GNN  (output dim = k_max; sliced to K_actual each forward call)
        self.s_conv = SAGEConv(in_channels, k_max)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _aux_losses(
        self,
        s: Tensor,          # (B*N, K)  soft assignment
        A_pool: Tensor,     # (B, K, K) pooled adjacency
        num_edges_per_graph: int,
        N: int,
        B: int,
        K: int,
    ) -> Dict[str, Tensor]:
        """
        Compute LP loss and entropy regularization without any N×N matrix.

        LP loss derivation
        ------------------
        ||A − SS^T||_F²
          = ||A||_F² − 2·<A, SS^T> + ||SS^T||_F²
          = |E|  − 2·tr(S^T A S)   + ||S^T S||_F²

        where:
          ||A||_F²   = |E|              (binary adjacency)
          tr(S^T A S) = tr(A_pool)      (trace is sum of diagonals)
          ||SS^T||_F² = ||S^T S||_F²   (cyclic trace identity)
        """
        # Link prediction loss
        tr_Apool  = A_pool.diagonal(dim1=1, dim2=2).sum(dim=1)      # (B,)
        S_3d      = s.view(B, N, K)
        STS       = torch.bmm(S_3d.transpose(1, 2), S_3d)           # (B, K, K)
        sts_frob2 = (STS * STS).sum(dim=(1, 2))                     # (B,)
        # Squared Frobenius form: ||A − SS^T||_F^2 / N^2.
        # Avoids sqrt(0) → ∞ gradient when the quantity hits zero, which
        # happens once cluster assignments improve and is the primary cause
        # of NaN parameter corruption in DiffPool training.
        lp_loss   = (num_edges_per_graph - 2.0 * tr_Apool + sts_frob2).clamp(min=0.0).mean() / (N * N)

        # Entropy regularization
        entropy_loss = (-s * torch.log(s.clamp(min=1e-8))).sum() / (B * N)

        return {"lp_loss": lp_loss, "entropy_loss": entropy_loss}

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,           # (B*N, in_channels)
        edge_index: Tensor,  # (2, E_total)  COO, already offset per-graph
        batch: Tensor,       # (B*N,)
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Returns
        -------
        x_new      : (B*K, hidden_channels)
        ei_new     : (2, E_new)  pooled edge_index
        batch_new  : (B*K,)
        aux_losses : {"lp_loss": scalar, "entropy_loss": scalar}
        """
        B       = int(batch.max().item()) + 1
        N_total = x.size(0)
        N       = N_total // B                           # nodes per graph (uniform)
        K       = min(int(N * self.ratio), self.k_max)  # actual cluster count

        # ── Z-GNN ─────────────────────────────────────────────────────────────
        z = F.leaky_relu(self.z_bn(self.z_conv(x, edge_index)), negative_slope=0.01)

        # ── S-GNN → soft assignment (N × K) ──────────────────────────────────
        s = torch.softmax(self.s_conv(x, edge_index)[:, :K], dim=-1)   # (B*N, K)

        # ── Sparse A @ S  (no dense N×N adjacency) ───────────────────────────
        row, col = edge_index
        AS = scatter(s[col], row, dim=0, dim_size=N_total, reduce='sum')  # (B*N, K)

        # ── Batch-aware pooling via bmm ───────────────────────────────────────
        S_3d  = s.view(B, N, K)                                  # (B, N, K)
        AS_3d = AS.view(B, N, K)                                  # (B, N, K)
        Z_3d  = z.view(B, N, -1)                                  # (B, N, hidden)

        x_new  = torch.bmm(S_3d.transpose(1, 2), Z_3d).reshape(B * K, -1)  # (B*K, h)
        A_pool = torch.bmm(S_3d.transpose(1, 2), AS_3d)                     # (B, K, K)

        # ── Build pooled edge_index from dense (B, K, K) A_pool ──────────────
        # A_pool is at most (B, 128, 128) = 0.5 MB — safe for dense ops.
        offsets       = torch.arange(B, device=x.device) * K
        b_idx, sk, dk = (A_pool > 1e-6).nonzero(as_tuple=True)
        ei_new        = torch.stack([sk + offsets[b_idx],
                                     dk + offsets[b_idx]])           # (2, E_new)

        batch_new = torch.arange(B, device=x.device).repeat_interleave(K)

        # ── Auxiliary losses ──────────────────────────────────────────────────
        num_edges = edge_index.size(1) // B
        aux = self._aux_losses(s, A_pool, num_edges, N, B, K)

        return x_new, ei_new, batch_new, aux


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalGraphSAGE(nn.Module):
    """
    GraphSAGE with two DiffPool layers.

    Reduces effective graph diameter before the final message-passing layer,
    addressing over-squashing caused by low spectral gap (small λ₁).

    API is backward-compatible with Phase 1 BaselineGraphSAGE: same positional
    arguments, but forward() now returns (scores, aux_losses) instead of scores.
    config_mask is accepted for compatibility but is unused — original node
    indices are lost after DiffPool; cluster embeddings are pooled globally.
    """

    def __init__(
        self,
        in_channels: int      = 158,
        hidden_channels: int  = 128,
        dropout: float        = 0.0,   # retained for API compat; not applied
        ratio: float          = 0.25,
    ) -> None:
        super().__init__()

        # ── Pre-pool GNN ──────────────────────────────────────────────────────
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.pre_conv   = SAGEConv(hidden_channels, hidden_channels)
        self.pre_bn     = nn.BatchNorm1d(hidden_channels, track_running_stats=False)

        # ── DiffPool layers ───────────────────────────────────────────────────
        self.pool1 = DiffPoolLayer(hidden_channels, hidden_channels,
                                   k_max=K1_MAX, ratio=ratio)
        self.pool2 = DiffPoolLayer(hidden_channels, hidden_channels,
                                   k_max=K2_MAX, ratio=ratio)

        # ── Post-pool GNNs ────────────────────────────────────────────────────
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels, track_running_stats=False)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3   = nn.BatchNorm1d(hidden_channels, track_running_stats=False)

        # ── MLP head ──────────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x: Tensor,            # (B*N, in_channels)
        edge_index: Tensor,   # (2, E_total)
        batch: Tensor,        # (B*N,)
        config_mask: Tensor,  # unused after DiffPool; kept for API compat
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Returns
        -------
        scores     : (B,)  one score per (graph, config) pair
        aux_losses : {"lp_loss": scalar, "entropy_loss": scalar}
        """
        # ── Input projection + pre-pool SAGEConv ──────────────────────────────
        x = F.leaky_relu(self.input_proj(x), negative_slope=0.01)
        h = F.leaky_relu(self.pre_bn(self.pre_conv(x, edge_index)), negative_slope=0.01)
        x = x + h

        # ── DiffPool 1 ────────────────────────────────────────────────────────
        x, edge_index, batch, aux1 = self.pool1(x, edge_index, batch)

        # ── SAGEConv after pool 1 (residual) ──────────────────────────────────
        h = F.leaky_relu(self.bn2(self.conv2(x, edge_index)), negative_slope=0.01)
        x = x + h

        # ── DiffPool 2 ────────────────────────────────────────────────────────
        x, edge_index, batch, aux2 = self.pool2(x, edge_index, batch)

        # ── SAGEConv after pool 2 (residual) ──────────────────────────────────
        h = F.leaky_relu(self.bn3(self.conv3(x, edge_index)), negative_slope=0.01)
        x = x + h

        # ── Readout ───────────────────────────────────────────────────────────
        graph_emb = global_mean_pool(x, batch)      # (B, hidden)
        scores    = self.mlp(graph_emb).squeeze(-1) # (B,)

        aux_losses = {
            "lp_loss":      aux1["lp_loss"]      + aux2["lp_loss"],
            "entropy_loss": aux1["entropy_loss"] + aux2["entropy_loss"],
        }
        return scores, aux_losses
