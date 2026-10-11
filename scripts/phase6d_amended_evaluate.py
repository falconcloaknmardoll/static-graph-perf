"""Phase 6d §10.5 (amended, exploratory) evaluation, exactly as frozen in
experiments/phase6d_nlp_default_preregistration.md §10.5 + Amendment 2 (§11).

Reads val_tau / epoch from each run's best_model.pt (§4 selection rule), then:
  A-P1  hardware@1e-4 - flat@1e-4        paired two-sided t, alpha 0.05, n = 5 (single primary)
  CC    hardware@1e-4 - flat@1e-3        confound check, same test; A-P1 is "supported" only if
                                         this also passes (§10.5)
  A-P2  sign of A-P1 vs xla:default (+) and nlp:random (-)   descriptive (§3 P2)
Reported alongside (not gates): Wilcoxon, bootstrap CI, Cohen's d_z, per-seed table, best epoch,
realized MDE, §10.5 collapse rule (val_n_graphs = 0 -> tau = 0), Amendment 2 completion check.
Every number is AMENDED / exploratory and never replaces §10.4 (registered training failure).
  python scripts/phase6d_amended_evaluate.py   -> experiments/phase6d_amended_results.json
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from power_analysis import min_detectable_effect  # noqa: E402

SEEDS = [42, 100, 123, 456, 789]
AMD = Path("checkpoints/expansion/nlp_default_lr1e-4")
EXP = Path("checkpoints/expansion/nlp_default")
LOGS = Path("logs/capped")


def run(path):
    c = torch.load(path / "best_model.pt", map_location="cpu", weights_only=False)
    t = float(c["val_tau"])
    a = c.get("args", {})
    return {"tau": 0.0 if math.isnan(t) else t, "epoch": int(c["epoch"]), "nan": math.isnan(t),
            "lr": a.get("lr"), "fast_eval": a.get("fast_eval"), "done": (path / ".done").exists()}


def n_graphs(log):
    m = re.findall(r"val_n_graphs (\d+)", (LOGS / log).read_text(errors="replace"))
    return int(m[-1]) if m else None


def contrast(a, b, rng):
    a, b = np.array(a), np.array(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return {"per_seed_delta": d.round(4).tolist(), "mean_delta": float(d.mean()),
            "sd_delta": float(d.std(ddof=1)), "t": float(t), "p": float(p),
            "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "bootstrap95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "cohens_dz": float(d.mean() / d.std(ddof=1)),
            "realized_mde": min_detectable_effect(float(d.std(ddof=1)), 5)}


def main():
    rng = np.random.default_rng(0)
    hw = [run(AMD / f"hardware-lr1e-4-seed{s}") for s in SEEDS]
    fl4 = [run(AMD / f"wide-baseline-lr1e-4-seed{s}") for s in SEEDS]
    fl3 = [run(EXP / f"wide-baseline-seed{s}") for s in SEEDS]
    all_done = all(r["done"] for r in hw + fl4 + fl3)

    ap1 = contrast([r["tau"] for r in hw], [r["tau"] for r in fl4], rng)
    cc = contrast([r["tau"] for r in hw], [r["tau"] for r in fl3], rng)

    # §7 read as A-P1 (amended), then the §10.5 confound check
    m, p = ap1["mean_delta"], ap1["p"]
    if m > 0 and p < 0.05:
        row = "A-P1 passes (Δτ > 0, p < .05)"
        verdict = ("A-P1 SUPPORTED (amended)" if cc["mean_delta"] > 0 and cc["p"] < 0.05
                   else "not supported — attributable to the flat arm's LR (§10.5 confound check fails)")
    elif m > 0:
        row, verdict = "Δτ > 0, p ≥ .05", "A-P1 NOT CONFIRMED (amended): directional only, quote MDE"
    elif m < 0:
        row, verdict = "Δτ < 0 (any p)", "A-P1 FAILS and A-P2 FAILS (amended): sign patterns with nlp:random"
    else:
        row, verdict = "Δτ ≈ 0", "A-P1 FAILS (amended)"

    out = {
        "label": "AMENDED / exploratory (6d §10.5). Never replaces §10.4.",
        "completion_all_13_done_before_cutoff": all_done,
        "arms": {name: {"per_seed": [round(r["tau"], 4) for r in v],
                        "mean": float(np.mean([r["tau"] for r in v])),
                        "sd": float(np.std([r["tau"] for r in v], ddof=1)),
                        "best_epoch": [r["epoch"] for r in v],
                        "lr_in_ckpt": [r["lr"] for r in v],
                        "fast_eval_in_ckpt": [r["fast_eval"] for r in v],
                        "nan_tau": [r["nan"] for r in v]}
                 for name, v in (("hardware@1e-4", hw), ("flat@1e-4", fl4), ("flat@1e-3", fl3))},
        "A_P1_hardware1e-4_minus_flat1e-4": {**ap1, "s7_row": row},
        "confound_check_hardware1e-4_minus_flat1e-3": cc,
        "verdict": verdict,
        "A_P2_sign_pattern": {"nlp_default_amended": "+" if m > 0 else "-",
                              "xla_default_registered_lr1e-3": "+", "nlp_random_lr1e-3": "-"},
        "collapse_check_val_n_graphs": {
            "hardware@1e-4": [n_graphs(f"nlp_default_hardware-lr1e-4_seed{s}.log") for s in SEEDS],
            "flat@1e-4": [n_graphs(f"nlp_default_wide-baseline-lr1e-4_seed{s}.log") for s in SEEDS],
            "flat@1e-3": [n_graphs(f"nlp_default_wide-baseline_seed{s}.log") for s in SEEDS]},
        "seeds": SEEDS,
    }
    Path("experiments/phase6d_amended_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
