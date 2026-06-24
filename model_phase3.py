"""
model_phase3.py
---------------
Phase 3 — HardwareAwareDiffPool: edge-aware hierarchical pooling.

What's new vs Phase 2 (`model_phase2.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 2c proved that DiffPool clustering on **data-flow edges alone** gives no
benefit over a parameter-matched flat GNN. Phase 3's thesis: the graph must also
carry **hardware-contention edges** (MXU / memory), and the model must actually
*consume* them. Two changes:

  1. **Edge-aware backbone.** Every `SAGEConv` becomes a `GINEConv`, which folds an
     edge feature into each message:  GINE updates node i as
         nn( (1+ε)·x_i  +  Σ_j ReLU(x_j + e_ij) ).
     A single shared `edge_encoder: Linear(3 → hidden)` projects our 3-D one-hot
     edge type [data, mxu, mem] up to the node dimension so it can be added inside
     the conv. Because the assignment GNN (S-GNN) is now edge-aware, **clustering
     can group nodes by contention, not just data flow** — exactly the lever
     Phase 2c showed was missing.

  2. **Edge coarsening.** When DiffPool merges nodes into clusters, the edges between
     clusters must be merged too — and their *types* summed. For clusters U, V the
     new 3-vector is
         EA_new[U, V, c] = Σ_{u,v} S[u,U] · E_c[u,v] · S[v,V]   =   Sᵀ Eᵤ S   (per channel)
     i.e. the soft-weighted count of original type-c edges crossing U↔V. In the hard
     limit, 3 data + 2 MXU edges between the two clusters → super-edge feature
     [3, 2, 0]. Computed sparsely (per-channel `scatter_add`), never materializing
     a dense N×N×3 tensor.

Design choice — keep the pooling *topology and losses* identical to Phase 2.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The binary adjacency A used for node coarsening (X_new = SᵀZ, A_pool = SᵀAS) and
for the LP / entropy auxiliary losses is computed *exactly* as in Phase 2 (each edge
counts once). The coarsened **edge attribute** rides alongside as extra information
fed to the next GINE layer; it does not alter the pooling math or the (validated)
loss derivation. So Phase 3 is a clean, controlled add-on to Phase 2 — the only new
signal is the edge types, which is the variable under test.

Parameter budget: ~236k at hidden=128, matched to Phase 2 / Phase 2c for a fair fight.

Architecture
~~~~~~~~~~~~
  Input (B*N, 158)
    → input_proj → GINEConv               [pre-pool, edge-aware, residual]
    → DiffPool 1  (nodes + edges)          [K1 = min(⌈0.25·N⌉, 128)]
    → GINEConv                             [post-pool-1, residual]
    → DiffPool 2  (nodes + edges)          [K2 = min(⌈0.25·K1⌉, 32)]
    → GINEConv                             [post-pool-2, residual]
    → global_mean_pool → MLP(→1)
  loss = listmle + 0.1·lp_loss + 0.01·entropy_loss   (same weights as Phase 2)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_scatter import scatter


K1_MAX: int = 128   # cluster budget after first DiffPool
K2_MAX: int = 32    # cluster budget after second DiffPool
EDGE_DIM: int = 3   # [data_flow, mxu_contention, mem_contention]


def _gine(in_dim: int, out_dim: int) -> GINEConv:
    """A GINEConv whose update MLP is Linear→BN→LeakyReLU→Linear.

    edge_dim is left at None: edge features are expected pre-projected to `in_dim`
    (we share one `edge_encoder` across the whole model). BN uses current-batch
    statistics (track_running_stats=False), consistent with the rest of the project.
    """
    mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, track_running_stats=False),
        nn.LeakyReLU(0.01),
        nn.Linear(out_dim, out_dim),
    )
    return GINEConv(mlp, train_eps=True)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-aware DiffPool layer
# ─────────────────────────────────────────────────────────────────────────────

class HardwareDiffPoolLayer(nn.Module):
    """One memory-safe DiffPool layer that coarsens BOTH nodes and edge attributes.

    Parameters
    ----------
    in_channels     : input node feature dim
    hidden_channels : output (cluster embedding) dim
    k_max           : hard cap on cluster count
    ratio           : target node-keep fraction (0.25)
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

        # Z-GNN (cluster embeddings) and S-GNN (soft assignment), both edge-aware.
        self.z_conv = _gine(in_channels, hidden_channels)
        self.s_conv = _gine(in_channels, k_max)   # output sliced to K each forward

    # ── auxiliary losses (identical math to Phase 2; binary adjacency) ──────────
    def _aux_losses(
        self,
        s: Tensor,           # (B*N, K)
        A_pool: Tensor,      # (B, K, K)  binary-derived
        num_edges_per_graph: float,
        N: int,
        B: int,
        K: int,
    ) -> Dict[str, Tensor]:
        tr_Apool  = A_pool.diagonal(dim1=1, dim2=2).sum(dim=1)       # (B,)
        S_3d      = s.view(B, N, K)
        STS       = torch.bmm(S_3d.transpose(1, 2), S_3d)            # (B, K, K)
        sts_frob2 = (STS * STS).sum(dim=(1, 2))                      # (B,)
        lp_loss   = (num_edges_per_graph - 2.0 * tr_Apool + sts_frob2).clamp(min=0.0).mean() / (N * N)
        entropy_loss = (-s * torch.log(s.clamp(min=1e-8))).sum() / (B * N)
        return {"lp_loss": lp_loss, "entropy_loss": entropy_loss}

    # ── forward ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: Tensor,            # (B*N, in_channels)
        edge_index: Tensor,   # (2, E_total)
        edge_attr_h: Tensor,  # (E_total, in_channels)  pre-projected edge features
        edge_attr_3: Tensor,  # (E_total, 3)            raw type multiplicities
        batch: Tensor,        # (B*N,)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Returns
        -------
        x_new          : (B*K, hidden)
        ei_new         : (2, E_new)
        edge_attr_new  : (E_new, 3)   coarsened type multiplicities
        batch_new      : (B*K,)
        aux_losses     : {"lp_loss", "entropy_loss"}
        """
        B       = int(batch.max().item()) + 1
        N_total = x.size(0)
        N       = N_total // B
        K       = min(int(N * self.ratio), self.k_max)

        # ── Z-GNN and S-GNN (edge-aware) ─────────────────────────────────────────
        z = F.leaky_relu(self.z_conv(x, edge_index, edge_attr_h), negative_slope=0.01)
        s = torch.softmax(self.s_conv(x, edge_index, edge_attr_h)[:, :K], dim=-1)  # (B*N, K)

        row, col = edge_index
        s_col = s[col]                                                   # (E, K)

        # ── Binary A @ S  (Phase 2 regime: each edge counts once) ────────────────
        AS = scatter(s_col, row, dim=0, dim_size=N_total, reduce='sum')  # (B*N, K)

        # ── Per-channel (E_c @ S) for edge-attribute coarsening ──────────────────
        AS_ch = [
            scatter(s_col * edge_attr_3[:, c:c + 1], row, dim=0,
                    dim_size=N_total, reduce='sum')                      # (B*N, K)
            for c in range(EDGE_DIM)
        ]

        # ── Batch-aware pooling via bmm ──────────────────────────────────────────
        S_3d  = s.view(B, N, K)
        Z_3d  = z.view(B, N, -1)
        AS_3d = AS.view(B, N, K)

        x_new  = torch.bmm(S_3d.transpose(1, 2), Z_3d).reshape(B * K, -1)  # (B*K, h)
        A_pool = torch.bmm(S_3d.transpose(1, 2), AS_3d)                    # (B, K, K) binary
        EA_pool = torch.stack([                                            # (B, K, K, 3)
            torch.bmm(S_3d.transpose(1, 2), AS_ch[c].view(B, N, K))
            for c in range(EDGE_DIM)
        ], dim=-1)

        # ── Build pooled edges from total cross-cluster multiplicity ─────────────
        EA_total      = EA_pool.sum(dim=-1)                               # (B, K, K)
        offsets       = torch.arange(B, device=x.device) * K
        b_idx, sk, dk = (EA_total > 1e-6).nonzero(as_tuple=True)
        ei_new        = torch.stack([sk + offsets[b_idx],
                                     dk + offsets[b_idx]])                # (2, E_new)
        edge_attr_new = EA_pool[b_idx, sk, dk]                            # (E_new, 3)
        batch_new     = torch.arange(B, device=x.device).repeat_interleave(K)

        # ── Auxiliary losses (binary adjacency, Phase 2 derivation) ──────────────
        num_edges = edge_index.size(1) / B
        aux = self._aux_losses(s, A_pool, num_edges, N, B, K)

        return x_new, ei_new, edge_attr_new, batch_new, aux


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class HardwareAwareDiffPool(nn.Module):
    """Edge-aware hierarchical GNN for TPUGraphs layout ranking (Phase 3).

    API mirrors Phase 2's `HierarchicalGraphSAGE` but `forward` takes an extra
    `edge_attr` (the (E, 3) one-hot edge types from `dataset.add_structural_edges`)
    and returns (scores, aux_losses).
    """

    def __init__(
        self,
        in_channels: int     = 158,
        hidden_channels: int = 128,
        dropout: float       = 0.0,   # API compat; unused
        ratio: float         = 0.25,
    ) -> None:
        super().__init__()

        # Shared projection of 3-D edge types → node dimension (used at every level).
        self.edge_encoder = nn.Linear(EDGE_DIM, hidden_channels)

        # Pre-pool backbone GINE.
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.pre_conv   = _gine(hidden_channels, hidden_channels)

        # DiffPool layers (coarsen nodes + edges).
        self.pool1 = HardwareDiffPoolLayer(hidden_channels, hidden_channels,
                                           k_max=K1_MAX, ratio=ratio)
        self.pool2 = HardwareDiffPoolLayer(hidden_channels, hidden_channels,
                                           k_max=K2_MAX, ratio=ratio)

        # Post-pool backbone GINEs.
        self.conv2 = _gine(hidden_channels, hidden_channels)
        self.conv3 = _gine(hidden_channels, hidden_channels)

        # MLP head.
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
        config_mask: Tensor,  # unused after pooling; kept for API compat
        edge_attr: Tensor,    # (E_total, 3)  one-hot edge types
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # ── Input projection + pre-pool GINE (residual) ──────────────────────────
        x   = F.leaky_relu(self.input_proj(x), negative_slope=0.01)
        e_h = self.edge_encoder(edge_attr)                                # level 0
        x   = x + F.leaky_relu(self.pre_conv(x, edge_index, e_h), negative_slope=0.01)

        # ── DiffPool 1 (nodes + edges) ───────────────────────────────────────────
        x, edge_index, edge_attr, batch, aux1 = self.pool1(
            x, edge_index, e_h, edge_attr, batch)

        # ── Post-pool-1 GINE (residual) ──────────────────────────────────────────
        e_h = self.edge_encoder(edge_attr)                                # level 1
        x   = x + F.leaky_relu(self.conv2(x, edge_index, e_h), negative_slope=0.01)

        # ── DiffPool 2 (nodes + edges) ───────────────────────────────────────────
        x, edge_index, edge_attr, batch, aux2 = self.pool2(
            x, edge_index, e_h, edge_attr, batch)

        # ── Post-pool-2 GINE (residual) ──────────────────────────────────────────
        e_h = self.edge_encoder(edge_attr)                                # level 2
        x   = x + F.leaky_relu(self.conv3(x, edge_index, e_h), negative_slope=0.01)

        # ── Readout ──────────────────────────────────────────────────────────────
        graph_emb = global_mean_pool(x, batch)
        scores    = self.mlp(graph_emb).squeeze(-1)

        aux_losses = {
            "lp_loss":      aux1["lp_loss"]      + aux2["lp_loss"],
            "entropy_loss": aux1["entropy_loss"] + aux2["entropy_loss"],
        }
        return scores, aux_losses
