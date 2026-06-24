#!/usr/bin/env python3
"""
phase4_evaluate_topk.py
-----------------------
Phase 4 — Downstream autotuner metric: Top-K runtime, not global Kendall's τ.

Kendall's τ scores the *entire* ranking. But a real XLA autotuner only ever uses
the model's top few predictions: it asks the model for its best guesses, measures
those on hardware, and keeps the fastest. So the question that actually matters is:

    "If I trust the model's top-K configs, how close is the best of them to the
     true optimal config for that graph?"

Two architectures (flat wide-baseline vs hardware-aware DiffPool) had statistically
indistinguishable global τ (Phase 2c / Phase 3). This script checks whether they
differ at the **extreme top** of the distribution, which is all the compiler cares about.

Metric — Top-K slowdown (lower is better; 1.0 = optimal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each graph, with the model's K highest-scored configs:

    slowdown@K = ( min actual runtime among the model's top-K configs )
                 ---------------------------------------------------------
                 (        min actual runtime over ALL configs           )

slowdown@K = 1.0 means the model's top-K already contains a config as fast as the
true best. slowdown@K = 1.20 means the autotuner, trusting this model's top-K, ends
up 20% slower than optimal. We report the mean over the 7 validation graphs.
(Equivalently reported as error% = (slowdown − 1) × 100.)

Prediction convention: the model is trained (ListMLE) so a HIGHER score = predicted
FASTER (lower runtime). Top-K = the K configs with the highest predicted scores.

Usage
-----
    python phase4_evaluate_topk.py                 # seed 42 pair (default)
    python phase4_evaluate_topk.py --seed 100
    python phase4_evaluate_topk.py --all-seeds      # mean over all 5 matched pairs
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from dataset import TPUGraphDataset
from model import BaselineGraphSAGE, BaselineGraphSAGE_Large
from model_phase2 import HierarchicalGraphSAGE
from model_phase3 import HardwareAwareDiffPool
from train import _forward

_DATA_ROOT = os.path.join(
    "data", "predict-ai-model-runtime", "npz_all", "npz", "layout", "xla", "random"
)
KS = (1, 5, 10)
SEEDS = (42, 100, 123, 456, 789)


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction from a checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def build_model(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Instantiate the right architecture from a checkpoint's saved args."""
    a = ckpt["args"]
    kind = a["model"]
    hidden = a.get("hidden", 128)
    if kind == "baseline":
        m = BaselineGraphSAGE(in_channels=158, hidden_channels=hidden, dropout=0.0)
    elif kind == "wide-baseline":
        m = BaselineGraphSAGE_Large(in_channels=158, dropout=0.0)
    elif kind == "hierarchical":
        m = HierarchicalGraphSAGE(in_channels=158, hidden_channels=hidden, dropout=0.0)
    elif kind == "hardware":
        m = HardwareAwareDiffPool(in_channels=158, hidden_channels=hidden, dropout=0.0)
    else:
        raise ValueError(f"unknown model kind in checkpoint: {kind!r}")
    m.load_state_dict(ckpt["model_state"])
    return m.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph predictions + Top-K
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def graph_preds(model, ds, file_idx, flat_indices, device, batch_size=32):
    """Return (pred_scores, true_runtimes) arrays aligned per config for one graph."""
    cfg_indices   = [ds._index[i][1] for i in flat_indices]
    npz           = ds._get_handle(file_idx)
    true_runtimes = np.array(npz["config_runtime"], dtype=np.float64)[cfg_indices]

    preds: List[torch.Tensor] = []
    for start in range(0, len(flat_indices), batch_size):
        chunk = flat_indices[start:start + batch_size]
        batch = Batch.from_data_list([ds[i] for i in chunk]).to(device)
        out   = _forward(model, batch)
        p     = out[0] if isinstance(out, tuple) else out
        preds.append(p.cpu())
    return torch.cat(preds).numpy(), true_runtimes


def topk_slowdowns(pred: np.ndarray, runtime: np.ndarray, ks=KS) -> Dict[int, float]:
    """slowdown@K = (best runtime in model's top-K) / (best runtime overall)."""
    best_overall = float(runtime.min())
    order = np.argsort(-pred)            # descending score = predicted fastest first
    out: Dict[int, float] = {}
    for k in ks:
        topk = order[:k]
        best_in_topk = float(runtime[topk].min())
        out[k] = best_in_topk / best_overall if best_overall > 0 else float("nan")
    return out


def evaluate_model(model, ds, device) -> Dict[int, float]:
    """Mean Top-K slowdown across all validation graphs (≥ max(K) configs)."""
    groups: Dict[int, List[int]] = defaultdict(list)
    for flat_idx, (file_idx, _) in enumerate(ds._index):
        groups[file_idx].append(flat_idx)

    per_k: Dict[int, List[float]] = {k: [] for k in KS}
    for file_idx, flat_indices in groups.items():
        if len(flat_indices) < max(KS):
            continue
        pred, runtime = graph_preds(model, ds, file_idx, flat_indices, device)
        sd = topk_slowdowns(pred, runtime)
        for k in KS:
            per_k[k].append(sd[k])
    return {k: float(np.mean(v)) for k, v in per_k.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def run_pair(seed: int, ds, device) -> Tuple[Dict[int, float], Dict[int, float]]:
    wide_ck = _ckpt(f"checkpoints/ablation/wide-baseline-seed{seed}/best_model.pt")
    hw_ck   = _ckpt(f"checkpoints/phase3/hardware-seed{seed}/best_model.pt")
    wide = evaluate_model(build_model(wide_ck, device), ds, device)
    hw   = evaluate_model(build_model(hw_ck,   device), ds, device)
    return wide, hw


def print_table(wide: Dict[int, float], hw: Dict[int, float], header: str) -> None:
    print("\n" + "═" * 60)
    print(f"  {header}")
    print(f"  Top-K slowdown vs optimal   (1.000 = perfect, lower = better)")
    print("═" * 60)
    print(f"  {'K':>3}   {'wide-baseline':>14}   {'hardware':>10}   {'Δ (hw−flat)':>12}")
    print(f"  {'─'*3}   {'─'*14}   {'─'*10}   {'─'*12}")
    for k in KS:
        d = hw[k] - wide[k]
        better = "hw better" if d < -1e-4 else ("flat better" if d > 1e-4 else "tie")
        print(f"  {k:>3}   {wide[k]:>14.4f}   {hw[k]:>10.4f}   {d:>+12.4f}  ({better})")
    print("═" * 60)
    print("  (error% = (slowdown − 1) × 100:  "
          f"K=1 flat {(wide[1]-1)*100:.1f}% / hw {(hw[1]-1)*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 — Top-K autotuner evaluation")
    ap.add_argument("--seed", type=int, default=42, help="matched seed pair to evaluate")
    ap.add_argument("--all-seeds", action="store_true",
                    help="average Top-K over all 5 matched pairs (robust to seed noise)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    val_ds = TPUGraphDataset(data_dir=os.path.join(_DATA_ROOT, "valid"))
    print(f"Valid : {val_ds.num_files} graphs")

    if not args.all_seeds:
        wide, hw = run_pair(args.seed, val_ds, device)
        print_table(wide, hw, f"Phase 4 — seed {args.seed} (single matched pair)")
        return

    # Average across all 5 matched pairs (honest, seed-noise-robust).
    acc_w = {k: [] for k in KS}
    acc_h = {k: [] for k in KS}
    for s in SEEDS:
        try:
            wide, hw = run_pair(s, val_ds, device)
        except FileNotFoundError as e:
            print(f"[skip] seed {s}: missing checkpoint ({e})")
            continue
        for k in KS:
            acc_w[k].append(wide[k]); acc_h[k].append(hw[k])
        print(f"  seed {s:>3}:  "
              f"flat@1={wide[1]:.4f} hw@1={hw[1]:.4f} | "
              f"flat@5={wide[5]:.4f} hw@5={hw[5]:.4f} | "
              f"flat@10={wide[10]:.4f} hw@10={hw[10]:.4f}")

    wide_mean = {k: float(np.mean(acc_w[k])) for k in KS}
    hw_mean   = {k: float(np.mean(acc_h[k])) for k in KS}
    wide_std  = {k: float(np.std(acc_w[k], ddof=1)) for k in KS}
    hw_std    = {k: float(np.std(acc_h[k], ddof=1)) for k in KS}
    print_table(wide_mean, hw_mean, f"Phase 4 — mean over {len(acc_w[1])} matched seed pairs")
    print("  per-K std (seed spread):")
    for k in KS:
        print(f"    K={k:>2}:  flat ±{wide_std[k]:.4f}   hw ±{hw_std[k]:.4f}")


if __name__ == "__main__":
    main()
