#!/usr/bin/env python3
"""
Phase 6b confound checks (PROJECT_TIMELINE.md A4):
  1. Per-graph tau breakdown on xla:default for flat (wide-baseline) and hardware models,
     across all 5 matched seeds -- is the aggregate win uniform across graphs or driven by
     a few outliers (bimodal)?
  2. Tie-structure of xla:default config_runtime arrays (model-independent) -- do excessive
     tied/near-identical runtimes explain a suppressed flat baseline?
  3. BH correction across the 3 pairwise Phase 6b comparisons (flat-vs-hier, hier-vs-hardware,
     flat-vs-hardware -- the third isn't in phase6b_summary.txt, computed here).

No new training. Loads existing checkpoints/expansion/xla_default/{wide-baseline,hardware,
hierarchical}-seed{N}/best_model.pt against the frozen xla:default valid split.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats
from torch_geometric.loader import DataLoader as _unused  # noqa
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import TPUGraphDataset, _forward
from model import BaselineGraphSAGE_Large
from model_phase2 import HierarchicalGraphSAGE
from model_phase3 import HardwareAwareDiffPool

SEEDS = [42, 100, 123, 456, 789]
CKPT = Path("checkpoints/expansion/xla_default")
VALID_DIR = "data/predict-ai-model-runtime/npz_all/npz/layout/xla/default/valid"


def per_graph_tau(model, val_ds, device, eval_batch_size=32):
    """Like train.evaluate() but returns {graph_name: tau} instead of just the mean."""
    model.eval()
    groups = defaultdict(list)
    for flat_idx, (file_idx, _) in enumerate(val_ds._index):
        groups[file_idx].append(flat_idx)

    out = {}
    for file_idx, flat_indices in groups.items():
        if len(flat_indices) < 2:
            continue
        name = Path(val_ds._file_paths[file_idx]).stem
        cfg_indices = [val_ds._index[i][1] for i in flat_indices]
        npz = val_ds._get_handle(file_idx)
        true_rt = np.array(npz["config_runtime"], dtype=np.float64)[cfg_indices]

        preds = []
        for start in range(0, len(flat_indices), eval_batch_size):
            chunk = flat_indices[start:start + eval_batch_size]
            data_list = [val_ds[i] for i in chunk]
            batch = Batch.from_data_list(data_list).to(device)
            with torch.no_grad():
                result = _forward(model, batch)
            p = result[0] if isinstance(result, tuple) else result
            preds.append(p.cpu())
        preds_np = torch.cat(preds).numpy()

        tau, _ = sp_stats.kendalltau(preds_np, -true_rt)
        if not np.isnan(tau):
            out[str(name)] = float(tau)
    return out


def load_model(kind, seed, device):
    ck = torch.load(CKPT / f"{kind}-seed{seed}" / "best_model.pt", map_location=device)
    if kind == "wide-baseline":
        m = BaselineGraphSAGE_Large(in_channels=158, dropout=0.0)
    elif kind == "hardware":
        m = HardwareAwareDiffPool(in_channels=158, hidden_channels=128, dropout=0.0)
    elif kind == "hierarchical":
        m = HierarchicalGraphSAGE(in_channels=158, hidden_channels=128, dropout=0.0)
    else:
        raise ValueError(kind)
    m.load_state_dict(ck["model_state"])
    return m.to(device)


def tie_structure(val_ds):
    """Model-independent: fraction of exactly-tied and near-tied (rel diff < 1%) runtime
    pairs per graph, from the raw config_runtime arrays. High tie fraction can suppress
    achievable Kendall's tau independent of model quality."""
    groups = defaultdict(list)
    for flat_idx, (file_idx, _) in enumerate(val_ds._index):
        groups[file_idx].append(flat_idx)

    out = {}
    for file_idx, flat_indices in groups.items():
        if len(flat_indices) < 2:
            continue
        name = Path(val_ds._file_paths[file_idx]).stem
        cfg_indices = [val_ds._index[i][1] for i in flat_indices]
        npz = val_ds._get_handle(file_idx)
        rt = np.array(npz["config_runtime"], dtype=np.float64)[cfg_indices]
        n = len(rt)
        if n > 4000:  # subsample for O(n^2) tie counting on huge graphs
            rng = np.random.default_rng(0)
            idx = rng.choice(n, 4000, replace=False)
            rt = rt[idx]
            n = 4000
        diffs = np.abs(rt[:, None] - rt[None, :])
        rel = diffs / (np.maximum(np.abs(rt[:, None]), np.abs(rt[None, :])) + 1e-12)
        iu = np.triu_indices(n, k=1)
        exact_tie_frac = float(np.mean(diffs[iu] == 0.0))
        near_tie_frac = float(np.mean(rel[iu] < 0.01))
        out[str(name)] = {"n_configs": int(n), "exact_tie_frac": exact_tie_frac,
                           "near_tie_1pct_frac": near_tie_frac}
    return out


def bh_correct(pvals):
    """Benjamini-Hochberg, returns adjusted p-values in original order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    val_ds = TPUGraphDataset(data_dir=VALID_DIR)
    print(f"valid: {val_ds.num_files} graphs, {len(val_ds):,} pairs", flush=True)

    # ── 1. Per-graph tau, flat + hardware, all 5 seeds ──────────────────────
    per_graph = {"wide-baseline": defaultdict(list), "hardware": defaultdict(list)}
    for kind in ["wide-baseline", "hardware"]:
        for s in SEEDS:
            m = load_model(kind, s, device)
            g2t = per_graph_tau(m, val_ds, device)
            for g, t in g2t.items():
                per_graph[kind][g].append(t)
            print(f"  {kind} seed{s}: {len(g2t)} graphs", flush=True)
            del m

    # ── 2. Tie structure (model-independent) ────────────────────────────────
    ties = tie_structure(val_ds)

    # ── 3. BH correction across the 3 Phase-6b pairwise tests ───────────────
    # Re-derive flat-vs-hardware directly here (flat-vs-hier and hier-vs-hardware are
    # already in phase6b_summary.txt; recompute all three fresh + BH-correct together).
    flat_means = [np.mean(per_graph["wide-baseline"][g]) for g in per_graph["wide-baseline"]]
    hw_means = [np.mean(per_graph["hardware"][g]) for g in per_graph["hardware"]]

    # Per-seed aggregate means (matches phase6b_summary.txt's per-seed table)
    flat_per_seed = {s: [] for s in SEEDS}
    hw_per_seed = {s: [] for s in SEEDS}
    for kind, per_seed in [("wide-baseline", flat_per_seed), ("hardware", hw_per_seed)]:
        for g, vals in per_graph[kind].items():
            for s, v in zip(SEEDS, vals):
                per_seed[s].append(v)
    flat_seed_means = np.array([np.mean(flat_per_seed[s]) for s in SEEDS])
    hw_seed_means = np.array([np.mean(hw_per_seed[s]) for s in SEEDS])

    t_flat_hw, p_flat_hw = sp_stats.ttest_rel(hw_seed_means, flat_seed_means)

    # flat-vs-hier and hier-vs-hardware come from the 3-arm GPU run already on disk
    # (checkpoints/expansion/xla_default/phase6b_summary.txt); flat-vs-hardware is the
    # fresh re-eval above. BH-correct all 3 together per PROJECT_TIMELINE.md A4.
    p_flat_hier, p_hier_hw = 0.0019, 0.0299
    labels = ["flat_vs_hier", "hier_vs_hardware", "flat_vs_hardware"]
    raw_p = [p_flat_hier, p_hier_hw, p_flat_hw]
    adj_p = bh_correct(raw_p)
    bh_out = {
        "inputs_raw_p": dict(zip(labels, raw_p)),
        "bh_adjusted_p": dict(zip(labels, adj_p)),
    }

    out = {
        "per_graph_tau": {k: dict(v) for k, v in per_graph.items()},
        "tie_structure": ties,
        "flat_seed_means": flat_seed_means.tolist(),
        "hw_seed_means": hw_seed_means.tolist(),
        "flat_vs_hardware_recomputed": {"t": float(t_flat_hw), "p": float(p_flat_hw)},
        "bh_correction_3way": bh_out,
    }
    Path("experiments/phase6b_confound_checks.json").write_text(json.dumps(out, indent=1))
    print("\nWrote experiments/phase6b_confound_checks.json")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n=== Per-graph tau (mean across 5 seeds) ===")
    print(f"{'graph':<45} {'flat':>8} {'hardware':>8} {'delta':>8}")
    for g in per_graph["wide-baseline"]:
        f = np.mean(per_graph["wide-baseline"][g])
        h = np.mean(per_graph["hardware"].get(g, [np.nan]))
        print(f"{g:<45} {f:>8.3f} {h:>8.3f} {h-f:>8.3f}")

    print("\n=== Tie structure (xla:default, model-independent) ===")
    print(f"{'graph':<45} {'n_configs':>10} {'exact_tie%':>11} {'near_tie(1%)%':>14}")
    for g, d in ties.items():
        print(f"{g:<45} {d['n_configs']:>10} {100*d['exact_tie_frac']:>10.2f}% "
              f"{100*d['near_tie_1pct_frac']:>13.2f}%")

    print(f"\nflat-vs-hardware (recomputed, 5 seeds): t={t_flat_hw:.4f}  p={p_flat_hw:.4f}")

    print("\n=== BH correction across 3 pairwise Phase 6b tests ===")
    for lbl in labels:
        print(f"  {lbl:<20} raw p={bh_out['inputs_raw_p'][lbl]:.6f}  "
              f"BH-adj p={bh_out['bh_adjusted_p'][lbl]:.6f}")


if __name__ == "__main__":
    main()
