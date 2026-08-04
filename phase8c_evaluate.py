#!/usr/bin/env python3
"""
phase8c_evaluate.py — pre-registered evaluation of the MXU co-design experiment.

Implements experiments/phase8c_preregistration.md §4/§7/§8 EXACTLY:
  • FROZEN split (§4): CONV = {inception_v3_batch_128_train, resnet50.4x4.fp16,
    resnet_v1_50_official_batch_128_bf16}; OTHER = the 4 bert/unet graphs.
    Evaluated on BOTH xla/default/valid and xla/random/valid → 6 conv instances,
    8 other instances.
  • Primary metric (§7): per-graph Top-1 slowdown (Phase-4 formula:
    runtime[argmax score] / min runtime, all configs of the graph), averaged
    into a per-seed CONV-subset mean; each feature arm compared to the baseline
    arm PAIRED BY SEED: paired t + Wilcoxon, bootstrap 95% CI on the paired
    difference, Cohen's d; Benjamini–Hochberg across the 3 feature arms.
  • Placebo (§7): identical test on the OTHER subset — must be ~unchanged.
  • Decision rules (§8) printed verbatim next to the numbers.

Perf: batches are built ONCE per (arm, collection, graph-chunk) and run through
all 5 seed-models of that arm (the Phase-5 lesson — batch building dominates,
BERT's 21k nodes especially). Raw predictions are cached per (arm, collection)
under checkpoints/phase8c/eval_cache/ so a crash never repeats finished work.

Usage
    python phase8c_evaluate.py                # everything
    python phase8c_evaluate.py --arms channel # subset (cache-friendly)
Outputs
    experiments/phase8c_results.md  +  .json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Batch

from dataset_mxu import MXUGraphDataset, ARM_COLUMNS
from model import BaselineGraphSAGE_Large
from train import _forward

SEEDS = [42, 100, 123, 456, 789]
ARMS = ["baseline", "channel", "spatial", "full"]
COLLECTIONS = ["xla/default", "xla/random"]
DATA = "data/predict-ai-model-runtime/npz_all/npz/layout"
CKPT = Path("checkpoints/phase8c/xla_random")
CACHE = Path("checkpoints/phase8c/eval_cache")

# FROZEN split — prereg §4 (rule mxu_frac ≥ 0.020). Do not edit.
CONV_GRAPHS = {
    "inception_v3_batch_128_train",
    "resnet50.4x4.fp16",
    "resnet_v1_50_official_batch_128_bf16",
}
OTHER_GRAPHS = {
    "bert_pretraining.4x4.fp16",
    "mlperf_bert_batch_24_2x2",
    "tf2_bert_pretrain_dynamic_batch_size",
    "unet_3d.4x4.bf16",
}


def load_arm_models(arm: str, device) -> Dict[int, torch.nn.Module]:
    models = {}
    for s in SEEDS:
        p = CKPT / f"mxu-{arm}-seed{s}" / "best_model.pt"
        ck = torch.load(p, map_location="cpu")
        assert ck.get("mxu_arm", arm) == arm, f"{p}: ckpt arm={ck.get('mxu_arm')} != {arm}"
        in_dim = ck.get("in_dim", 158 + len(ARM_COLUMNS[arm]))
        m = BaselineGraphSAGE_Large(in_channels=in_dim, dropout=0.0)
        m.load_state_dict(ck["model_state"])
        m.to(device).eval()
        models[s] = m
    return models


@torch.no_grad()
def arm_collection_slowdowns(arm: str, coll: str, device,
                             batch_size: int = 32) -> Dict[str, Dict[int, float]]:
    """{graph_name: {seed: top1_slowdown}} for one (arm, collection).

    Builds each eval batch ONCE and runs all 5 seed-models on it.
    Cached to CACHE/<arm>_<coll>.json.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"{arm}_{coll.replace('/', '_')}.json"
    if cpath.exists():
        return {g: {int(s): v for s, v in d.items()}
                for g, d in json.loads(cpath.read_text()).items()}

    ds = MXUGraphDataset(f"{DATA}/{coll}/valid", mxu_arm=arm)
    models = load_arm_models(arm, device)

    groups: Dict[int, List[int]] = defaultdict(list)
    for flat, (fi, _) in enumerate(ds._index):
        groups[fi].append(flat)

    out: Dict[str, Dict[int, float]] = {}
    for fi, flats in groups.items():
        name = Path(ds._file_paths[fi]).stem
        if name not in CONV_GRAPHS | OTHER_GRAPHS:
            continue
        npz = ds._get_handle(fi)
        cfg_idx = [ds._index[i][1] for i in flats]
        runtime = np.asarray(npz["config_runtime"], dtype=np.float64)[cfg_idx]

        preds = {s: [] for s in SEEDS}
        for start in range(0, len(flats), batch_size):
            chunk = flats[start:start + batch_size]
            batch = Batch.from_data_list([ds[i] for i in chunk]).to(device)
            for s, m in models.items():                      # batch built once, 5 models
                r = _forward(m, batch)
                preds[s].append((r[0] if isinstance(r, tuple) else r).cpu())

        out[name] = {}
        best = float(runtime.min())
        for s in SEEDS:
            p = torch.cat(preds[s]).numpy()
            out[name][s] = float(runtime[int(np.argmax(p))] / best)
        print(f"    {arm:<9} {coll:<12} {name[:38]:<40} "
              + "  ".join(f"{out[name][s]:.3f}" for s in SEEDS), flush=True)

    cpath.write_text(json.dumps(out))
    return out


def bh_correct(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg adjusted p-values (monotone step-up)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in list(enumerate(order))[::-1]:
        prev = min(prev, pvals[i] * m / (rank + 1))
        adj[i] = prev
    return adj.tolist()


def paired_stats(arm_vals: List[float], base_vals: List[float]) -> dict:
    from scipy import stats as st
    d = np.array(arm_vals) - np.array(base_vals)          # negative = arm improves
    t, p_t = st.ttest_rel(arm_vals, base_vals)
    try:
        _, p_w = st.wilcoxon(arm_vals, base_vals)
    except ValueError:
        p_w = float("nan")
    rng = np.random.default_rng(0)
    boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    cohend = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    return {"delta": float(d.mean()), "p_t": float(p_t), "p_w": float(p_w),
            "ci": [float(lo), float(hi)], "d": cohend}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="experiments/phase8c_results.md",
                    help="output md path; use a separate file for control runs "
                         "so the frozen primary result is not overwritten")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}   arms={args.arms}")

    # ── raw per-graph slowdowns ────────────────────────────────────────────
    raw: Dict[str, dict] = {}          # arm -> {(coll,graph): {seed: slowdown}}
    for arm in args.arms:
        raw[arm] = {}
        for coll in COLLECTIONS:
            for g, sv in arm_collection_slowdowns(arm, coll, device,
                                                  args.batch_size).items():
                raw[arm][f"{coll}::{g}"] = sv

    # ── per-seed subset means ──────────────────────────────────────────────
    def subset_means(arm: str, names: set) -> List[float]:
        vals = []
        for s in SEEDS:
            xs = [sv[s] for key, sv in raw[arm].items()
                  if key.split("::")[1] in names]
            vals.append(float(np.mean(xs)))
        return vals

    lines: List[str] = []
    def P(t=""): print(t); lines.append(t)

    P("# Phase 8c — pre-registered results (frozen split, paired by seed)\n")
    P(f"Instances: CONV = 3 graphs × 2 collections, OTHER = 4 × 2.\n")
    P(f"{'arm':<10} {'CONV top-1 slowdown':>22} {'OTHER top-1 slowdown':>23}")
    per_arm = {}
    for arm in args.arms:
        cv, ov = subset_means(arm, CONV_GRAPHS), subset_means(arm, OTHER_GRAPHS)
        per_arm[arm] = {"conv": cv, "other": ov}
        P(f"{arm:<10} {np.mean(cv):>12.4f} ± {np.std(cv, ddof=1):.4f}"
          f" {np.mean(ov):>13.4f} ± {np.std(ov, ddof=1):.4f}")

    results = {"per_arm_seed_means": per_arm, "raw": raw}
    if "baseline" in per_arm:
        feature_arms = [a for a in args.arms if a != "baseline"]
        P("\n## Primary test (§7): CONV subset, arm vs baseline, paired by seed")
        prim = {a: paired_stats(per_arm[a]["conv"], per_arm["baseline"]["conv"])
                for a in feature_arms}
        adj = bh_correct([prim[a]["p_t"] for a in feature_arms])
        for a, padj in zip(feature_arms, adj):
            r = prim[a]; r["p_bh"] = padj
            P(f"  {a:<9} Δslowdown={r['delta']:+.4f}  CI95=[{r['ci'][0]:+.4f},"
              f"{r['ci'][1]:+.4f}]  t-p={r['p_t']:.4f}  BH-p={padj:.4f}  "
              f"Wilcoxon-p={r['p_w']:.3f}  d={r['d']:+.2f}")
        P("\n## Placebo (§7): OTHER subset (must be ~unchanged)")
        plac = {a: paired_stats(per_arm[a]["other"], per_arm["baseline"]["other"])
                for a in feature_arms}
        for a in feature_arms:
            r = plac[a]
            P(f"  {a:<9} Δslowdown={r['delta']:+.4f}  CI95=[{r['ci'][0]:+.4f},"
              f"{r['ci'][1]:+.4f}]  t-p={r['p_t']:.4f}")
        results["primary"] = prim
        results["placebo"] = plac
        P("\n## Decision rules (§8, verbatim)")
        P("  gap closes (BH-p<0.05, OTHER unchanged) → positive co-design result")
        P("  gap unchanged → representational-floor evidence (publishable)")
        P("  OTHER moves too → confound; investigate before any claim")
        P("  mixed across arms → report per-arm; +spatial vs +channel contrast IS the finding (P1/P2)")
        P("\n  Registered predictions: P1 channel→null, P2 spatial→best-but-uncertain,")
        P("  P3 persisting gap = reportable finding. (Committed 2026-06-29, before data.)")

    # ── P4 specificity control (prereg A2): channel vs scramble, paired ──────
    if "channel" in per_arm and "scramble" in per_arm:
        P("\n## Specificity control (prereg A2): CONV subset, channel vs scramble, paired by seed")
        P("  P4: channel significantly < scramble ⇒ real MXU info; channel ≈ scramble ⇒ capacity artifact.")
        r = paired_stats(per_arm["channel"]["conv"], per_arm["scramble"]["conv"])
        P(f"  channel−scramble  Δslowdown={r['delta']:+.4f}  CI95=[{r['ci'][0]:+.4f},"
          f"{r['ci'][1]:+.4f}]  t-p={r['p_t']:.4f}  Wilcoxon-p={r['p_w']:.3f}  d={r['d']:+.2f}")
        rp = paired_stats(per_arm["scramble"]["other"], per_arm["baseline"]["other"])
        P(f"  placebo (scramble−baseline, OTHER)  Δ={rp['delta']:+.4f}  t-p={rp['p_t']:.4f}")
        results["control_p4"] = {"channel_vs_scramble": r, "scramble_placebo": rp}

    # JSON follows --out so a control run cannot clobber the frozen primary result.
    out_md = Path(args.out)
    out_json = out_md.with_suffix(".json")
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(results, indent=2, default=float))
    P(f"\nWrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
