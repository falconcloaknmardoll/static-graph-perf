#!/usr/bin/env python3
"""
phase5_error_analysis.py
------------------------
Phase 5 — Hardware Bottleneck Attribution (error analysis).

Goal: the models leave ~21-25% on the table at Top-1 (Phase 4). WHERE are they
blind? We isolate the worst-mispredicted graphs (Top-1 pick furthest from optimal)
and ask whether they are structurally unusual — opcode mix, configurable-node
density, spectral gap λ₁, diameter — relative to the typical dataset graph.

⚠️  READ THIS FIRST — the sample-size honesty guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The held-out validation set has **7 graphs**. "Top 10% most mispredicted" is
therefore **0-1 graphs**. You cannot make a statistically confident architectural
claim ("models fail on graphs with Feature X") from one graph — that is exactly the
single-sample trap this entire project exists to expose (see VERIFICATION.md C9:
seed variance and tiny-sample noise dominate everything here).

So this script is deliberately built as **exploratory / hypothesis-generating**, not
confirmatory. It:
  • computes per-graph Top-1 misprediction on the 7 held-out val graphs (mean over
    the 5 seeds, for BOTH models — never a single seed),
  • characterizes the structural features of the worst graph(s) vs the dataset norm,
  • reports Spearman correlations (feature vs misprediction) WITH their p-values and a
    blunt note that at n=7, |ρ|≳0.78 is needed for p<0.05,
  • REFUSES to print a confident causal claim; it prints what is needed to earn one
    (the official/larger TEST set, or many more held-out graphs).

Reuses verified components: analyze_topology.compute_spectral_gap (λ₁),
analyze_opcodes.compute_config_spread, dataset.MXU_OPS/MEM_OPS (corrected enum),
phase4_evaluate_topk for predictions.

Usage
-----
    python phase5_error_analysis.py                  # val mispred + features
    python phase5_error_analysis.py --max-train 25    # cap dataset-norm sample
    python phase5_error_analysis.py --recompute        # ignore cached mispredictions
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import scipy.stats as sp_stats
import torch
from torch_geometric.data import Batch

from dataset import TPUGraphDataset, MXU_OPS, MEM_OPS
from analyze_opcodes import (
    build_adjacency_list, compute_config_spread, classify_opcode,
)
from analyze_topology import (
    pyg_to_undirected_networkx, pyg_to_symmetric_csr, _lcc_subgraph,
    compute_spectral_gap, compute_approx_diameter,
)
from phase4_evaluate_topk import build_model, topk_slowdowns, SEEDS
from train import _forward

_DATA_ROOT = os.path.join(
    "data", "predict-ai-model-runtime", "npz_all", "npz", "layout", "xla", "random"
)
_CACHE = os.path.join("checkpoints", "phase5_mispred.json")


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph structural features (computed on the RAW data-flow graph, not augmented)
# ─────────────────────────────────────────────────────────────────────────────

def graph_features(npz_path: str, diam_sweeps: int = 5) -> Optional[Dict[str, float]]:
    try:
        npz = np.load(npz_path, mmap_mode="r")
        opcodes    = np.array(npz["node_opcode"],     dtype=np.int64)
        edge_index = np.array(npz["edge_index"].T,    dtype=np.int64)   # (2, E) data-flow
        config_ids = np.array(npz["node_config_ids"], dtype=np.int64)
        N = int(opcodes.shape[0])

        # opcode mix (corrected TPUGraphs enum, via dataset.MXU_OPS / MEM_OPS)
        mxu_frac = float(np.isin(opcodes, list(MXU_OPS)).mean())
        mem_frac = float(np.isin(opcodes, list(MEM_OPS)).mean())
        labels = np.array([classify_opcode(int(o)) for o in opcodes])
        compute_ratio = float((labels == "compute").mean())
        memory_ratio  = float((labels == "memory").mean())

        # configurable-node density (median pairwise hop distance; ÷diameter)
        adj = build_adjacency_list(edge_index, N)
        med_dist, frac_close = compute_config_spread(adj, config_ids)

        # topology: spectral gap λ₁ on the LCC + approx diameter
        G = pyg_to_undirected_networkx(edge_index, N)
        A = pyg_to_symmetric_csr(edge_index, N)
        G_lcc, A_lcc, lcc_frac = _lcc_subgraph(G, A)
        lambda_1, _ = compute_spectral_gap(A_lcc, k=6)
        diameter = compute_approx_diameter(G_lcc, n_sweeps=diam_sweeps)

        norm_spread = med_dist / max(diameter, 1)
        return {
            "n_nodes": float(N),
            "mxu_frac": mxu_frac,
            "mem_frac": mem_frac,
            "compute_ratio": compute_ratio,
            "memory_ratio": memory_ratio,
            "median_config_dist": med_dist,
            "norm_config_spread": norm_spread,
            "spectral_gap": float(lambda_1),
            "diameter": float(diameter),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] features failed on {os.path.basename(npz_path)}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph misprediction on the held-out val set (mean over seeds, both models)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_graph_mispred(val_ds, device, recompute: bool = False, batch_size: int = 32) -> Dict[str, dict]:
    """Returns {graph_filename: {flat_top1, hw_top1, mean_top1, ...}} averaged over seeds.

    Efficiency: the Early-Join feature construction (`val_ds[i]`) is the dominant cost,
    so we build each minibatch ONCE and run all 10 models (5 seeds × {flat, hw}) on it,
    instead of rebuilding the data per model. ~10× fewer Early-Join constructions.
    """
    if os.path.isfile(_CACHE) and not recompute:
        with open(_CACHE) as f:
            print(f"[cache] loaded per-graph misprediction from {_CACHE}")
            return json.load(f)

    # Preload all 10 models once (tiny: 234k params each).
    models: Dict[tuple, torch.nn.Module] = {}
    for seed in SEEDS:
        models[(seed, "flat")] = build_model(torch.load(
            f"checkpoints/ablation/wide-baseline-seed{seed}/best_model.pt",
            map_location="cpu"), device)
        models[(seed, "hw")] = build_model(torch.load(
            f"checkpoints/phase3/hardware-seed{seed}/best_model.pt",
            map_location="cpu"), device)

    groups: Dict[int, List[int]] = defaultdict(list)
    for flat_idx, (file_idx, _) in enumerate(val_ds._index):
        groups[file_idx].append(flat_idx)

    out: Dict[str, dict] = {}
    for fi, flat_indices in groups.items():
        if len(flat_indices) < 10:
            continue
        cfg_indices = [val_ds._index[i][1] for i in flat_indices]
        npz = val_ds._get_handle(fi)
        runtime = np.array(npz["config_runtime"], dtype=np.float64)[cfg_indices]

        # collect per-model predictions, building each batch only once
        preds: Dict[tuple, List[torch.Tensor]] = {key: [] for key in models}
        for start in range(0, len(flat_indices), batch_size):
            chunk = flat_indices[start:start + batch_size]
            batch = Batch.from_data_list([val_ds[i] for i in chunk]).to(device)
            for key, model in models.items():
                o = _forward(model, batch)
                p = o[0] if isinstance(o, tuple) else o
                preds[key].append(p.cpu())

        # per-seed slowdowns → average over seeds, per model tag
        acc = {"flat": defaultdict(list), "hw": defaultdict(list)}
        for (seed, tag), chunks in preds.items():
            sd = topk_slowdowns(torch.cat(chunks).numpy(), runtime)
            for k, v in sd.items():
                acc[tag][k].append(v)

        fname = os.path.basename(val_ds._file_paths[fi])
        rec = {}
        for k in (1, 5, 10):
            rec[f"flat_top{k}"] = float(np.mean(acc["flat"][k]))
            rec[f"hw_top{k}"]   = float(np.mean(acc["hw"][k]))
            rec[f"mean_top{k}"] = float(np.mean(acc["flat"][k] + acc["hw"][k]))
        out[fname] = rec
        print(f"  {fname[:46]:<46} flat@1={rec['flat_top1']:.3f} hw@1={rec['hw_top1']:.3f}")

    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[cache] wrote {_CACHE}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ["n_nodes", "mxu_frac", "mem_frac", "compute_ratio", "memory_ratio",
            "median_config_dist", "norm_config_spread", "spectral_gap", "diameter"]
FMT = {"n_nodes": "{:.0f}", "diameter": "{:.0f}", "median_config_dist": "{:.1f}"}


def _fmt(feat: str, v: float) -> str:
    return FMT.get(feat, "{:.4f}").format(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 — hardware bottleneck attribution")
    ap.add_argument("--max-train", type=int, default=25,
                    help="sample this many train graphs for the dataset-norm baseline")
    ap.add_argument("--recompute", action="store_true", help="ignore cached mispredictions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Misprediction on held-out val graphs (mean over 5 seeds) ──────────────
    val_ds = TPUGraphDataset(data_dir=os.path.join(_DATA_ROOT, "valid"))
    print(f"Val: {val_ds.num_files} graphs   (⚠️  top-10% = {max(1, math.ceil(0.1*val_ds.num_files))} graph)")
    mis = per_graph_mispred(val_ds, device, recompute=args.recompute)

    # ── 2. Features for val graphs + a dataset-norm sample from train ────────────
    val_files = {os.path.basename(p): p for p in glob.glob(
        os.path.join(_DATA_ROOT, "valid", "*.npz"))}
    print("\nComputing structural features (val graphs)…")
    val_feat = {fn: graph_features(val_files[fn]) for fn in mis if fn in val_files}
    val_feat = {k: v for k, v in val_feat.items() if v is not None}

    train_paths = sorted(glob.glob(os.path.join(_DATA_ROOT, "train", "*.npz")))
    rng = np.random.default_rng(0)
    if len(train_paths) > args.max_train:
        train_paths = list(rng.choice(train_paths, size=args.max_train, replace=False))
    print(f"Computing dataset-norm features ({len(train_paths)} train graphs sampled)…")
    train_feat = [f for p in train_paths if (f := graph_features(p)) is not None]

    # ── 3. Rank val graphs by mean Top-1 misprediction ──────────────────────────
    ranked = sorted(val_feat.keys(), key=lambda fn: mis[fn]["mean_top1"], reverse=True)
    n_worst = max(1, math.ceil(0.1 * len(ranked)))
    worst = ranked[:n_worst]

    # ── 4. Comparative table ─────────────────────────────────────────────────────
    def col(stats_list, feat):
        vals = [s[feat] for s in stats_list]
        return np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else float("nan"))

    print("\n" + "═" * 78)
    print("  PHASE 5 — structural features:  dataset norm  vs  worst-mispredicted")
    print("═" * 78)
    print(f"  {'feature':<20} {'dataset (train)':>18} {'all val':>14} {'WORST val 10%':>16}")
    print(f"  {'─'*20} {'─'*18} {'─'*14} {'─'*16}")
    ds_stats = train_feat
    val_stats = list(val_feat.values())
    worst_stats = [val_feat[fn] for fn in worst]
    for feat in FEATURES:
        dm, ds_sd = col(ds_stats, feat)
        vm, _     = col(val_stats, feat)
        wm, _     = col(worst_stats, feat)
        z = (wm - dm) / ds_sd if ds_sd and not math.isnan(ds_sd) and ds_sd > 0 else float("nan")
        flag = "  ←" if abs(z) >= 1.5 else ""
        print(f"  {feat:<20} {_fmt(feat, dm):>18} {_fmt(feat, vm):>14} "
              f"{_fmt(feat, wm):>16}  (z={z:+.1f}){flag}")

    print("\n  worst-mispredicted graph(s), by mean(flat,hw) Top-1 slowdown:")
    for fn in worst:
        print(f"    {fn[:48]:<48}  flat@1={mis[fn]['flat_top1']:.3f}  hw@1={mis[fn]['hw_top1']:.3f}")

    # ── 5. Spearman: feature vs misprediction across the 7 val graphs ────────────
    print("\n  Spearman ρ (feature vs mean Top-1 slowdown across val graphs):")
    n = len(val_feat)
    slow = np.array([mis[fn]["mean_top1"] for fn in val_feat])
    print(f"    {'feature':<20} {'ρ':>7} {'p':>8}   (n={n}; need |ρ|≳0.78 for p<0.05)")
    for feat in FEATURES:
        x = np.array([val_feat[fn][feat] for fn in val_feat])
        rho, p = sp_stats.spearmanr(x, slow)
        star = " *" if p < 0.05 else ""
        print(f"    {feat:<20} {rho:>+7.3f} {p:>8.3f}{star}")

    # ── 6. The honesty guard — no fabricated claim ───────────────────────────────
    print("\n" + "═" * 78)
    print("  INTERPRETATION GUARD (read before quoting anything above)")
    print("═" * 78)
    print(f"""  • n = {n} held-out graphs. The 'WORST val 10%' column is {n_worst} graph(s).
    A z-score or a single graph's feature values CANNOT support a confident claim
    of the form 'models fail on graphs with Feature X'. This is the same
    small-sample trap documented project-wide (VERIFICATION.md C9).
  • Spearman ρ above is descriptive. At n={n}, only |ρ|≳0.78 reaches p<0.05, and even
    that is one correlation among {len(FEATURES)} tested (no multiple-comparison control).
  • Treat any pattern here as a HYPOTHESIS to be tested on the official/larger TEST
    set (more held-out graphs), not as a finding. To earn the architectural claim
    'Hardware resource Y is too non-deterministic for static features', you would
    additionally want a non-determinism measure (e.g. runtime variance among
    near-identical configs) — structure alone cannot establish non-determinism.
  • What this script legitimately delivers: a reproducible PIPELINE + an honest
    exploratory look, not a confirmatory result.""")
    print("═" * 78)


if __name__ == "__main__":
    main()
