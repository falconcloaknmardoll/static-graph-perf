"""Phase 6f evaluation, exactly as frozen in experiments/phase6f_hardware_lr_preregistration.md.

Reads val_tau / epoch from each run's best_model.pt (§4 selection rule), picks LR*_hw over the
enlarged grid {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} (boundary step fired: LR*_hw was 1e-4, a grid edge),
takes LR*_flat from phase6e_results.json unchanged, then:
  Q1  hardware@LR*_hw - flat@LR*_flat   paired two-sided t, alpha 0.05, single primary (§5)
  Q2  hardware@LR*_hw - hardware@1e-3   descriptive, no test
Reported alongside (not gates): both LR curves, Wilcoxon, bootstrap CI, Cohen's d_z, per-seed
table, best epoch per run, realized MDE, and the §4 convergence check (best epoch >= 46).
  python scripts/phase6f_evaluate.py   -> experiments/phase6f_results.json
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
GRID = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]
EXP = Path("checkpoints/expansion/xla_default")
FLATLR = Path("checkpoints/flatlr/xla_default")
HWLR = Path("checkpoints/hwlr/xla_default")
LOGS = Path("logs/capped")


def run(path):
    c = torch.load(path / "best_model.pt", map_location="cpu", weights_only=False)
    t = float(c["val_tau"])
    return (0.0 if math.isnan(t) else t), int(c["epoch"])   # §4 collapse rule: constant output -> 0


def hw_dir(lr, s):
    return EXP / f"hardware-seed{s}" if lr == "1e-3" else HWLR / f"hardware-lr{lr}-seed{s}"


def flat_dir(lr, s):
    return EXP / f"wide-baseline-seed{s}" if lr == "1e-3" else FLATLR / f"wide-baseline-lr{lr}-seed{s}"


def n_graphs(lr, s):
    # final evaluate()'s val_n_graphs from the wandb summary in the run log (new runs only)
    m = re.findall(r"val_n_graphs (\d+)", (LOGS / f"xla_default_hardware-lr{lr}_seed{s}.log").read_text())
    return int(m[-1]) if m else None


def main():
    rng = np.random.default_rng(0)
    hw = {lr: [run(hw_dir(lr, s)) for s in SEEDS] for lr in GRID}
    hw_tau = {lr: [t for t, _ in v] for lr, v in hw.items()}

    means = {lr: float(np.mean(v)) for lr, v in hw_tau.items()}
    best = max(means.values())
    tied = [lr for lr in GRID if means[lr] == best]
    lr_hw = "1e-3" if "1e-3" in tied else tied[0]        # §4: exact tie goes to 1e-3

    e6 = json.loads(Path("experiments/phase6e_results.json").read_text())
    lr_flat = e6["lr_star"]                               # §4: taken from 6e unchanged
    flat = [run(flat_dir(lr_flat, s)) for s in SEEDS]
    flat_tau = [t for t, _ in flat]

    a, b = np.array(hw_tau[lr_hw]), np.array(flat_tau)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    if p < 0.05:
        verdict = "structure helps under symmetric tuning" if d.mean() > 0 else "tuned flat beats tuned structure"
    else:
        verdict = "no difference detected under symmetric tuning (n = 5)"

    conv = {"hardware": sum(e >= 46 for _, e in hw[lr_hw]), "flat": sum(e >= 46 for _, e in flat)}
    budget_qualifier = conv["hardware"] >= 3 or conv["flat"] >= 3

    out = {
        "hardware_curve": {lr: {"mean": means[lr], "sd": float(np.std(hw_tau[lr], ddof=1)),
                                "per_seed": [round(x, 4) for x in hw_tau[lr]],
                                "best_epoch": [e for _, e in hw[lr]]} for lr in GRID},
        "flat_curve_from_6e": {lr: e6["flat_curve"][lr]["mean"] for lr in GRID},
        "lr_star_hw": lr_hw,
        "lr_star_hw_at_edge_of_enlarged_grid": lr_hw in (GRID[0], GRID[-1]),
        "lr_star_flat": lr_flat,
        "flat_at_lr_star": {"per_seed": [round(x, 4) for x in flat_tau], "best_epoch": [e for _, e in flat]},
        "Q1_hardware_at_lr_star_hw_minus_flat_at_lr_star_flat": {
            "per_seed_delta": d.round(4).tolist(),
            "mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
            "t": float(t), "p": float(p),
            "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "bootstrap95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "cohens_dz": float(d.mean() / d.std(ddof=1)),
            "realized_mde": min_detectable_effect(float(d.std(ddof=1)), 5),
            "verdict": verdict + (" -- within a 50-epoch budget" if budget_qualifier else ""),
        },
        "Q2_hardware_at_lr_star_hw_minus_hardware_at_1e-3": means[lr_hw] - means["1e-3"],
        "convergence_best_epoch_ge_46": conv,
        "within_50_epoch_budget_qualifier": budget_qualifier,
        "collapse_check_val_n_graphs": {lr: [n_graphs(lr, s) for s in SEEDS] for lr in GRID if lr != "1e-3"},
        "seeds": SEEDS,
    }
    Path("experiments/phase6f_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
