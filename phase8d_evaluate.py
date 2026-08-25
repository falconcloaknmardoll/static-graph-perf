#!/usr/bin/env python3
"""
phase8d_evaluate.py — pre-registered evaluation of the 8d oracle capstone probe.

Implements experiments/phase8d_preregistration.md §7/§8/§9 EXACTLY, reusing
phase8c's machinery/split/protocol unchanged (prereg §5):
  • FROZEN split (shared with 8c): CONV = {inception_v3_batch_128_train,
    resnet50.4x4.fp16, resnet_v1_50_official_batch_128_bf16}; OTHER = the 4
    bert/unet graphs. Evaluated on BOTH xla/default/valid and xla/random/valid.
  • baseline and channel arms are REUSED UNCHANGED from 8c — their per-graph
    Top-1 slowdowns are read straight out of
    checkpoints/phase8c/eval_cache/{baseline,channel}_{coll}.json (no new
    inference; those checkpoints/caches are untouched by 8d).
  • oracle-bucket is the one NEW arm: fresh inference over
    checkpoints/phase8d/xla_random/oracle-bucket-seed*/, using
    dataset_oracle.OracleGraphDataset (158+5 dims).
  • Primary test (§8): oracle-bucket vs channel, CONV subset, paired by seed.
  • Secondary: oracle-bucket vs baseline (full achievable gain).
  • Placebo (§8): OTHER subset, oracle-bucket vs baseline — NOT required to be
    null (the bucket signal is informative everywhere by construction); the
    discipline check is CONV gain >= OTHER gain.
  • P5 (registered before data): oracle-bucket CONV Top-1 slowdown <= 1.05.
  • Ceiling-recovery fraction: (baseline-channel)/(baseline-oracle).

Usage
    python phase8d_evaluate.py
Outputs
    experiments/phase8d_results.md  +  .json
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

from dataset_oracle import OracleGraphDataset
from model import BaselineGraphSAGE_Large
from train import _forward

SEEDS = [42, 100, 123, 456, 789]
COLLECTIONS = ["xla/default", "xla/random"]
DATA = "data/predict-ai-model-runtime/npz_all/npz/layout"
C8C_CACHE = Path("checkpoints/phase8c/eval_cache")
CKPT_8D = Path("checkpoints/phase8d/xla_random")

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


def load_8c_cache(arm: str, coll: str) -> Dict[str, Dict[int, float]]:
    """{graph_name: {seed: top1_slowdown}}, straight from 8c's finished cache."""
    cpath = C8C_CACHE / f"{arm}_{coll.replace('/', '_')}.json"
    if not cpath.exists():
        raise FileNotFoundError(
            f"{cpath} missing — 8c eval cache required for arm={arm!r}; "
            "run phase8c_evaluate.py first (should already be done)."
        )
    return {g: {int(s): v for s, v in d.items()}
            for g, d in json.loads(cpath.read_text()).items()}


def load_oracle_models(device) -> Dict[int, torch.nn.Module]:
    models = {}
    for s in SEEDS:
        p = CKPT_8D / f"oracle-bucket-seed{s}" / "best_model.pt"
        ck = torch.load(p, map_location="cpu")
        in_dim = ck.get("in_dim", 163)
        m = BaselineGraphSAGE_Large(in_channels=in_dim, dropout=0.0)
        m.load_state_dict(ck["model_state"])
        m.to(device).eval()
        models[s] = m
    return models


@torch.no_grad()
def oracle_collection_slowdowns(coll: str, device, batch_size: int = 32,
                                cache_dir: Path = Path("checkpoints/phase8d/eval_cache"),
                                ) -> Dict[str, Dict[int, float]]:
    """{graph_name: {seed: top1_slowdown}} for oracle-bucket on one collection."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"oracle-bucket_{coll.replace('/', '_')}.json"
    if cpath.exists():
        return {g: {int(s): v for s, v in d.items()}
                for g, d in json.loads(cpath.read_text()).items()}

    ds = OracleGraphDataset(f"{DATA}/{coll}/valid")
    models = load_oracle_models(device)

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
            for s, m in models.items():
                r = _forward(m, batch)
                preds[s].append((r[0] if isinstance(r, tuple) else r).cpu())

        out[name] = {}
        best = float(runtime.min())
        for s in SEEDS:
            p = torch.cat(preds[s]).numpy()
            out[name][s] = float(runtime[int(np.argmax(p))] / best)
        print(f"    oracle-bucket {coll:<12} {name[:38]:<40} "
              + "  ".join(f"{out[name][s]:.3f}" for s in SEEDS), flush=True)

    cpath.write_text(json.dumps(out))
    return out


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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="experiments/phase8d_results.md")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ── raw per-graph slowdowns: baseline+channel from 8c's cache, oracle fresh ──
    raw: Dict[str, dict] = {"baseline": {}, "channel": {}, "oracle-bucket": {}}
    for coll in COLLECTIONS:
        for g, sv in load_8c_cache("baseline", coll).items():
            raw["baseline"][f"{coll}::{g}"] = sv
        for g, sv in load_8c_cache("channel", coll).items():
            raw["channel"][f"{coll}::{g}"] = sv
        for g, sv in oracle_collection_slowdowns(coll, device, args.batch_size).items():
            raw["oracle-bucket"][f"{coll}::{g}"] = sv

    def subset_means(arm: str, names: set) -> List[float]:
        vals = []
        for s in SEEDS:
            xs = [sv[s] for key, sv in raw[arm].items()
                  if key.split("::")[1] in names]
            vals.append(float(np.mean(xs)))
        return vals

    lines: List[str] = []
    def P(t=""): print(t); lines.append(t)

    P("# Phase 8d — pre-registered oracle capstone results (frozen split, paired by seed)\n")
    P("Instances: CONV = 3 graphs x 2 collections, OTHER = 4 x 2. baseline/channel reused")
    P("unchanged from checkpoints/phase8c/eval_cache/ (no new inference for those two arms).\n")
    per_arm = {}
    P(f"{'arm':<14} {'CONV top-1 slowdown':>22} {'OTHER top-1 slowdown':>23}")
    for arm in ["baseline", "channel", "oracle-bucket"]:
        cv, ov = subset_means(arm, CONV_GRAPHS), subset_means(arm, OTHER_GRAPHS)
        per_arm[arm] = {"conv": cv, "other": ov}
        P(f"{arm:<14} {np.mean(cv):>12.4f} ± {np.std(cv, ddof=1):.4f}"
          f" {np.mean(ov):>13.4f} ± {np.std(ov, ddof=1):.4f}")

    results = {"per_arm_seed_means": per_arm, "raw": raw}

    P("\n## Primary test (prereg §8): oracle-bucket vs channel, CONV subset, paired by seed")
    prim = paired_stats(per_arm["oracle-bucket"]["conv"], per_arm["channel"]["conv"])
    P(f"  oracle-bucket vs channel  Δslowdown={prim['delta']:+.4f}  "
      f"CI95=[{prim['ci'][0]:+.4f},{prim['ci'][1]:+.4f}]  t-p={prim['p_t']:.4f}  "
      f"Wilcoxon-p={prim['p_w']:.3f}  d={prim['d']:+.2f}")

    P("\n## Secondary: oracle-bucket vs baseline, CONV subset (full achievable gain)")
    sec = paired_stats(per_arm["oracle-bucket"]["conv"], per_arm["baseline"]["conv"])
    P(f"  oracle-bucket vs baseline  Δslowdown={sec['delta']:+.4f}  "
      f"CI95=[{sec['ci'][0]:+.4f},{sec['ci'][1]:+.4f}]  t-p={sec['p_t']:.4f}  "
      f"Wilcoxon-p={sec['p_w']:.3f}  d={sec['d']:+.2f}")

    P("\n## Placebo (prereg §8): OTHER subset, oracle-bucket vs baseline")
    P("  NOT required to be null (bucket signal is informative everywhere by construction).")
    P("  Discipline check: CONV gain must be >= OTHER gain, or the conv-specific-floor")
    P("  story is undercut and must be reported honestly.")
    plac = paired_stats(per_arm["oracle-bucket"]["other"], per_arm["baseline"]["other"])
    P(f"  oracle-bucket vs baseline (OTHER)  Δslowdown={plac['delta']:+.4f}  "
      f"CI95=[{plac['ci'][0]:+.4f},{plac['ci'][1]:+.4f}]  t-p={plac['p_t']:.4f}")
    conv_gain = -sec["delta"]      # positive = improvement (slowdown went down)
    other_gain = -plac["delta"]
    P(f"  CONV gain={conv_gain:+.4f}  OTHER gain={other_gain:+.4f}  "
      f"CONV>=OTHER: {conv_gain >= other_gain}")

    results["primary_vs_channel"] = prim
    results["secondary_vs_baseline"] = sec
    results["placebo_other"] = plac
    results["conv_gain"] = conv_gain
    results["other_gain"] = other_gain
    results["conv_gain_ge_other_gain"] = bool(conv_gain >= other_gain)

    conv_oracle_mean = float(np.mean(per_arm["oracle-bucket"]["conv"]))
    conv_baseline_mean = float(np.mean(per_arm["baseline"]["conv"]))
    conv_channel_mean = float(np.mean(per_arm["channel"]["conv"]))
    ceiling_recovered = ((conv_baseline_mean - conv_channel_mean) /
                          (conv_baseline_mean - conv_oracle_mean)
                          if conv_baseline_mean != conv_oracle_mean else float("nan"))
    p5_pass = conv_oracle_mean <= 1.05
    results["conv_oracle_mean"] = conv_oracle_mean
    results["ceiling_recovered_by_channel"] = ceiling_recovered
    results["p5_pass"] = bool(p5_pass)

    P("\n## Registered prediction P5 (committed 2026-08-19, before data)")
    P(f"  oracle-bucket CONV Top-1 slowdown = {conv_oracle_mean:.4f}   "
      f"threshold <=1.05   PASS={p5_pass}")
    P(f"  Ceiling-recovery fraction (baseline-channel)/(baseline-oracle) = "
      f"{ceiling_recovered:.4f}")

    P("\n## Decision rules (prereg §9, verbatim)")
    P("  oracle<=1.05 & sig. beats channel -> P5 confirmed: post-8c residual is representational")
    P("  oracle beats channel but >1.05    -> partial: headroom exists, ceiling lower than hoped")
    P("  oracle ~= channel (no sig. gain)  -> sharper finding: residual is a floor (label noise /")
    P("                                        non-determinism, C10), not missing representation")
    P("  OTHER gain >= CONV gain           -> confound-adjacent: bucket signal not conv-specific")

    out_md = Path(args.out)
    out_json = out_md.with_suffix(".json")
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(results, indent=2, default=float))
    P(f"\nWrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
