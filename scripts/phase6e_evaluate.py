"""Phase 6e evaluation, exactly as frozen in experiments/phase6e_flat_lr_preregistration.md.

Reads val_tau from each run's best_model.pt (§4 selection rule), picks LR* over the enlarged
grid {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} (boundary step fired: LR* was 1e-4, a grid edge), then:
  P1  hardware@1e-3 - flat@LR*      paired two-sided t, unadjusted p decides (§5)
  P2  hierarchical@1e-3 - flat@LR*  paired t, Holm-adjusted jointly with P1
Reported alongside (not gates): flat LR curve, Wilcoxon, bootstrap CI, Cohen's d_z, per-seed
table, realized MDE, and P3 (flat@LR* vs the published 0.12).
  python scripts/phase6e_evaluate.py   -> experiments/phase6e_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from power_analysis import min_detectable_effect  # noqa: E402

SEEDS = [42, 100, 123, 456, 789]
GRID = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]
EXP = Path("checkpoints/expansion/xla_default")
FLATLR = Path("checkpoints/flatlr/xla_default")


def val_tau(path):
    t = float(torch.load(path / "best_model.pt", map_location="cpu", weights_only=False)["val_tau"])
    return 0.0 if math.isnan(t) else t           # §4 collapse rule: constant output -> tau = 0


def flat_dir(lr, s):
    return EXP / f"wide-baseline-seed{s}" if lr == "1e-3" else FLATLR / f"wide-baseline-lr{lr}-seed{s}"


def mde(sd, n=5):
    # the project's own registered MDE (paired t, alpha 0.05, power 0.8), so numbers match §6
    return min_detectable_effect(sd, n)


def contrast(a, b, rng):
    d = np.array(a) - np.array(b)
    t, p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return {
        "per_seed_delta": d.round(4).tolist(),
        "mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
        "t": float(t), "p": float(p),
        "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "bootstrap95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "cohens_dz": float(d.mean() / d.std(ddof=1)),
        "realized_mde": mde(float(d.std(ddof=1))),
    }


def main():
    rng = np.random.default_rng(0)
    flat = {lr: [val_tau(flat_dir(lr, s)) for s in SEEDS] for lr in GRID}
    hier = [val_tau(EXP / f"hierarchical-seed{s}") for s in SEEDS]
    hw = [val_tau(EXP / f"hardware-seed{s}") for s in SEEDS]

    means = {lr: float(np.mean(v)) for lr, v in flat.items()}
    best = max(means.values())
    tied = [lr for lr in GRID if means[lr] == best]
    lr_star = "1e-3" if "1e-3" in tied else tied[0]      # §4: exact tie goes to 1e-3

    p1 = contrast(hw, flat[lr_star], rng)
    p2 = contrast(hier, flat[lr_star], rng)
    ps = sorted([("P1", p1["p"]), ("P2", p2["p"])], key=lambda x: x[1])   # Holm, m = 2
    holm, running = {}, 0.0
    for i, (name, p) in enumerate(ps):
        running = max(running, min(1.0, (2 - i) * p))
        holm[name] = running
    p1["p_holm"], p2["p_holm"] = holm["P1"], holm["P2"]

    def verdict(c):
        if c["mean_delta"] <= 0:
            return "Delta <= 0: FAILS"
        if c["p"] >= 0.05:
            return "Delta > 0, p >= 0.05: not shown robust to baseline tuning"
        return "passes" + (" (LR* = 1e-3: robust)" if lr_star == "1e-3" else " (LR* != 1e-3: robust, smaller)")

    out = {
        "flat_curve": {lr: {"mean": means[lr], "sd": float(np.std(flat[lr], ddof=1)),
                            "per_seed": [round(x, 4) for x in flat[lr]]} for lr in GRID},
        "lr_star": lr_star,
        "lr_star_at_edge_of_enlarged_grid": lr_star in (GRID[0], GRID[-1]),
        "hardware_at_1e-3": [round(x, 4) for x in hw],
        "hierarchical_at_1e-3": [round(x, 4) for x in hier],
        "P1_hardware_minus_flat_at_lr_star": p1 | {"verdict": verdict(p1)},
        "P2_hierarchical_minus_flat_at_lr_star": p2 | {"verdict": verdict(p2)},
        "P3_flat_at_lr_star_vs_published_0.12": means[lr_star],
        "seeds": SEEDS,
    }
    Path("experiments/phase6e_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
