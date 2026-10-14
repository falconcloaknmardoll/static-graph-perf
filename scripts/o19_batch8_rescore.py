"""O19 — 8c/8d CONV Top-1 re-scored at eval batch 8 (exploratory sensitivity, eval-only).

Declared in experiments/o19_8c8d_batch8_declaration.md (commit before running). Existing
checkpoints only: 8c baseline/channel/spatial/full/scramble (checkpoints/phase8c/xla_random) and
8d oracle-bucket (checkpoints/phase8d/xla_random), 5 seeds each. Same split, Top-1 definition
and statistics as phase8c_evaluate.py / phase8d_evaluate.py; only the eval batch changes.
The registered batch-32 numbers are read from the original caches (never rewritten).

  python scripts/o19_batch8_rescore.py   -> experiments/o19_8c8d_batch8_results.{md,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase8c_evaluate as e8c                                         # noqa: E402
import phase8d_evaluate as e8d                                         # noqa: E402
from phase8c_evaluate import (CONV_GRAPHS, OTHER_GRAPHS, COLLECTIONS, SEEDS,  # noqa: E402
                              paired_stats, bh_correct)
from power_analysis import min_detectable_effect                      # noqa: E402

ARMS_8C = ["baseline", "channel", "spatial", "full", "scramble"]
REG_8C = Path("checkpoints/phase8c/eval_cache")       # registered batch-32 caches (read-only)
REG_8D = Path("checkpoints/phase8d/eval_cache")
NEW = Path("checkpoints/o19_eval_cache")              # every new cache lives here, batch-tagged
OUT = Path("experiments/o19_8c8d_batch8_results.md")


def read_cache(p):
    return {g: {int(s): v for s, v in d.items()} for g, d in json.loads(p.read_text()).items()}


def score_8c(arm, coll, bs, dev):
    """phase8c_evaluate.arm_collection_slowdowns, with its cache redirected to NEW/8c_b<bs>.
    The original function names its cache file without the batch size, so it must never
    point at REG_8C here."""
    e8c.CACHE = NEW / f"8c_b{bs}"
    assert e8c.CACHE.resolve() != REG_8C.resolve()
    return e8c.arm_collection_slowdowns(arm, coll, dev, batch_size=bs)


def score_8d(coll, bs, dev):
    cache = NEW / f"8d_b{bs}"
    assert cache.resolve() != REG_8D.resolve()
    return e8d.oracle_collection_slowdowns(coll, dev, batch_size=bs, cache_dir=cache)


def subset_means(raw, names):
    """Per-seed mean Top-1 slowdown over the instances (both collections) in `names`."""
    return [float(np.mean([sv[s] for key, sv in raw.items() if key.split("::")[1] in names]))
            for s in SEEDS]


def stats(a, b):
    ps = paired_stats(a, b)
    d = np.array(a) - np.array(b)
    ps["mde"] = min_detectable_effect(float(d.std(ddof=1)), len(SEEDS)) if d.std(ddof=1) > 0 else float("nan")
    ps["per_seed_delta"] = d.round(4).tolist()
    return ps


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    raw = {32: {}, 8: {}}                      # batch -> arm -> {"coll::graph": {seed: slowdown}}

    # 1. Reproduction gate: re-run batch 32 into NEW and compare with the registered caches.
    # baseline (cache Jul 13), scramble (Aug 19, GPU outage) and oracle (Sep 21) span the
    # periods whose caches may come from different hardware/software.
    repro = {}
    for name in ["baseline", "scramble", "oracle"]:
        for coll in COLLECTIONS:
            c = coll.replace('/', '_')
            if name == "oracle":
                reg, new = read_cache(REG_8D / f"oracle-bucket_{c}.json"), score_8d(coll, 32, dev)
            else:
                reg, new = read_cache(REG_8C / f"{name}_{c}.json"), score_8c(name, coll, 32, dev)
            diffs = [abs(new[g][s] - reg[g][s]) for g in reg for s in SEEDS]
            repro[f"{name} {coll}"] = {"cells": len(diffs), "cells_differing": int(sum(x > 1e-9 for x in diffs)),
                                       "max_abs_diff": float(max(diffs))}
            print(f"repro {name} {coll}: {repro[f'{name} {coll}']}", flush=True)

    # 2. Registered batch-32 numbers (original caches) and new batch-8 numbers.
    for arm in ARMS_8C:
        raw[32][arm], raw[8][arm] = {}, {}
        for coll in COLLECTIONS:
            for g, sv in read_cache(REG_8C / f"{arm}_{coll.replace('/', '_')}.json").items():
                raw[32][arm][f"{coll}::{g}"] = sv
            for g, sv in score_8c(arm, coll, 8, dev).items():
                raw[8][arm][f"{coll}::{g}"] = sv
    raw[32]["oracle"], raw[8]["oracle"] = {}, {}
    for coll in COLLECTIONS:
        for g, sv in read_cache(REG_8D / f"oracle-bucket_{coll.replace('/', '_')}.json").items():
            raw[32]["oracle"][f"{coll}::{g}"] = sv
        for g, sv in score_8d(coll, 8, dev).items():
            raw[8]["oracle"][f"{coll}::{g}"] = sv

    # 3. Declared contrasts (§3 of the declaration), at both batches.
    res = {"reproduction_gate": repro, "means": {}, "contrasts": {}, "raw": {str(b): raw[b] for b in raw}}
    L = ["# O19 — 8c/8d CONV Top-1 at eval batch 8 vs 32 (exploratory sensitivity)", "",
         "Declared in `experiments/o19_8c8d_batch8_declaration.md`. Batch 32 = registered numbers "
         "(original caches); batch 8 = this re-score. Δ < 0 means the first arm has lower slowdown.", "",
         "## Reproduction gate (batch 32 re-run vs registered cache)", ""]
    for k, r in repro.items():
        L.append(f"- {k}: {r['cells_differing']}/{r['cells']} graph×seed cells differ, max |Δ| {r['max_abs_diff']:.2e}")
    L += ["", "## Per-arm means (mean ± SD over seeds)", "",
          "| arm | CONV b32 | CONV b8 | OTHER b32 | OTHER b8 |", "|---|---|---|---|---|"]
    m = {b: {a: {"conv": subset_means(raw[b][a], CONV_GRAPHS), "other": subset_means(raw[b][a], OTHER_GRAPHS)}
             for a in raw[b]} for b in raw}
    res["means"] = {str(b): m[b] for b in m}
    for a in ARMS_8C + ["oracle"]:
        f = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"
        L.append(f"| {a} | {f(m[32][a]['conv'])} | {f(m[8][a]['conv'])} | {f(m[32][a]['other'])} | {f(m[8][a]['other'])} |")

    tests = [  # (key, first arm, second arm, subset, BH family or None)
        ("8c primary: channel − baseline", "channel", "baseline", "conv", "8c"),
        ("8c primary: spatial − baseline", "spatial", "baseline", "conv", "8c"),
        ("8c primary: full − baseline", "full", "baseline", "conv", "8c"),
        ("8c placebo: channel − baseline", "channel", "baseline", "other", None),
        ("8c placebo: spatial − baseline", "spatial", "baseline", "other", None),
        ("8c placebo: full − baseline", "full", "baseline", "other", None),
        ("P4: channel − scramble", "channel", "scramble", "conv", None),
        ("P4 placebo: scramble − baseline", "scramble", "baseline", "other", None),
        ("8d primary: oracle − channel", "oracle", "channel", "conv", None),
        ("8d secondary: oracle − baseline", "oracle", "baseline", "conv", None),
        ("8d placebo: oracle − baseline", "oracle", "baseline", "other", None),
    ]
    L += ["", "## Contrasts (paired by seed; two-sided t, Wilcoxon, bootstrap CI, BH over the 3 8c primary arms)", "",
          "| contrast | subset | b32 Δ | b32 p (BH) | b8 Δ | b8 p (BH) | b8 CI95 | b8 MDE | same verdict? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b in (32, 8):
        for key, a1, a2, sub, fam in tests:
            res["contrasts"][f"b{b}:{key}:{sub}"] = stats(m[b][a1][sub], m[b][a2][sub])
        prim = [f"b{b}:{k}:{s}" for k, _, _, s, fam in tests if fam == "8c"]
        for k, padj in zip(prim, bh_correct([res["contrasts"][k]["p_t"] for k in prim])):
            res["contrasts"][k]["p_bh"] = padj
    for key, _, _, sub, fam in tests:
        r32, r8 = res["contrasts"][f"b32:{key}:{sub}"], res["contrasts"][f"b8:{key}:{sub}"]
        p = lambda r: r.get("p_bh", r["p_t"])
        same = (np.sign(r32["delta"]) == np.sign(r8["delta"])) and ((p(r32) < .05) == (p(r8) < .05))
        L.append(f"| {key} | {sub.upper()} | {r32['delta']:+.4f} | {p(r32):.3g} | {r8['delta']:+.4f} | {p(r8):.3g} | "
                 f"[{r8['ci'][0]:+.4f}, {r8['ci'][1]:+.4f}] | {r8['mde']:.4f} | {'yes' if same else '**NO**'} |")

    oc = {b: float(np.mean(m[b]["oracle"]["conv"])) for b in m}
    rec = {b: (np.mean(m[b]["baseline"]["conv"]) - np.mean(m[b]["channel"]["conv"]))
              / (np.mean(m[b]["baseline"]["conv"]) - oc[b]) for b in m}
    res["p5"] = {str(b): {"oracle_conv": oc[b], "pass_le_1.05": oc[b] <= 1.05, "recovery": float(rec[b])} for b in m}
    L += ["", "## 8d descriptives", "",
          f"- Oracle CONV Top-1 (P5 threshold ≤ 1.05): b32 {oc[32]:.4f}, b8 {oc[8]:.4f}",
          f"- Ceiling-recovery fraction (baseline−channel)/(baseline−oracle): b32 {rec[32]:.4f}, b8 {rec[8]:.4f}"]

    OUT.write_text("\n".join(L) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L), f"\nWrote {OUT} and {OUT.with_suffix('.json')}", sep="\n")


if __name__ == "__main__":
    main()
