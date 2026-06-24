#!/usr/bin/env python3
"""
scripts/plot_ablation_results.py
--------------------------------
Visualise the parameter-matched ablation (Phase 2c): wide flat baseline vs
DiffPool hierarchical, 5 matched seeds.  Reads the best τ from each run's
checkpoint (the source of truth), so re-running regenerates from real results.

Outputs (into experiments/figures/):
  ablation_paired.png   per-seed paired scatter + group mean ± std

Run:
    python scripts/plot_ablation_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "experiments" / "figures"
CKPT_BASE = ROOT / "checkpoints" / "ablation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 100, 123, 456, 789]


def _tau(run_dir: str, base: Path = CKPT_BASE) -> float:
    import torch
    ckpt = base / run_dir / "best_model.pt"
    d = torch.load(ckpt, map_location="cpu")
    return float(d["val_tau"])


def plot_paired() -> None:
    wide = np.array([_tau(f"wide-baseline-seed{s}") for s in SEEDS])
    diff = np.array([_tau(f"diffpool-seed{s}") for s in SEEDS])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # per-seed paired lines (each seed = one faint connector)
    x_wide, x_diff = 0, 1
    for w, d, s in zip(wide, diff, SEEDS):
        color = "tab:green" if d > w else "tab:red"
        ax.plot([x_wide, x_diff], [w, d], "-", color=color, alpha=0.45, zorder=1)
        ax.scatter([x_wide, x_diff], [w, d], color=color, s=35, zorder=2)
    ax.text(x_diff + 0.04, diff[-1], f"  green: DiffPool wins\n  red: flat wins",
            fontsize=7.5, va="center", color="gray")

    # group means ± std (value label placed to the OUTER side of each diamond)
    for x, vals, dx, ha in [(x_wide, wide, -0.10, "right"),
                            (x_diff, diff, +0.10, "left")]:
        ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt="D", color="black",
                    markersize=9, capsize=6, lw=2, zorder=3)
        ax.text(x + dx, vals.mean(),
                f"{vals.mean():.3f}\n±{vals.std(ddof=1):.3f}",
                ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.axhline(0.34, color="crimson", linestyle=":", linewidth=1.3,
               label="TPUGraphs paper baseline (0.34)")
    ax.set_xticks([x_wide, x_diff])
    ax.set_xticklabels(["wide-baseline\n(flat, h=172)", "diffpool\n(hierarchical)"])
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylabel("Best validation Kendall's τ")
    d = diff - wide
    ax.set_title(
        "Parameter-matched ablation — does hierarchy help?\n"
        f"Δτ = {d.mean():+.4f} (paired t p=0.22, Wilcoxon p=0.31) — "
        "no significant difference",
        fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "ablation_paired.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def plot_phase3_paired() -> None:
    """Hardware (Phase 3) vs wide-baseline (Phase 2c), paired by seed."""
    p3 = ROOT / "checkpoints" / "phase3"
    flat = np.array([_tau(f"wide-baseline-seed{s}") for s in SEEDS])           # ablation/
    hw   = np.array([_tau(f"hardware-seed{s}", base=p3) for s in SEEDS])       # phase3/

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x_flat, x_hw = 0, 1
    for f, h in zip(flat, hw):
        color = "tab:green" if h > f else "tab:red"
        ax.plot([x_flat, x_hw], [f, h], "-", color=color, alpha=0.45, zorder=1)
        ax.scatter([x_flat, x_hw], [f, h], color=color, s=35, zorder=2)
    ax.text(x_hw + 0.04, hw.min(), "  green: hardware wins\n  red: flat wins",
            fontsize=7.5, va="center", color="gray")

    for x, vals, dx, ha in [(x_flat, flat, -0.10, "right"), (x_hw, hw, +0.10, "left")]:
        ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt="D", color="black",
                    markersize=9, capsize=6, lw=2, zorder=3)
        ax.text(x + dx, vals.mean(), f"{vals.mean():.3f}\n±{vals.std(ddof=1):.3f}",
                ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.axhline(0.34, color="crimson", linestyle=":", linewidth=1.3,
               label="TPUGraphs paper baseline (0.34)")
    ax.set_xticks([x_flat, x_hw])
    ax.set_xticklabels(["wide-baseline\n(flat, Phase 2c)", "hardware\n(edge-aware, Phase 3)"])
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylabel("Best validation Kendall's τ")
    d = hw - flat
    ax.set_title(
        "Phase 3 — do hardware-contention edges help?\n"
        f"Δτ = {d.mean():+.4f} (paired t p=0.72, Wilcoxon p=0.63) — "
        "no significant difference",
        fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "phase3_paired.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    plot_paired()
    plot_phase3_paired()
