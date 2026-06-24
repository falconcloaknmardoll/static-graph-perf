#!/usr/bin/env python3
"""
scripts/plot_training_results.py
--------------------------------
Generate the training-result figures for the experiment write-ups, parsed
*directly* from the canonical wandb run logs (no hardcoded numbers — re-running
this regenerates the figures from the source of truth).

Outputs (into experiments/figures/):
  training_curves.png     val τ vs epoch for all 4 runs (one panel)
  loss_equilibrium.png    listmle / lp / entropy vs epoch for DiffPool cold start

Run:
    python scripts/plot_training_results.py
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Canonical runs: (wandb id, human label, is_hierarchical)
RUNS = [
    ("s8cx8q38", "Phase 1 · GraphSAGE cold",        False),
    ("5b9ryxid", "Phase 2 · GraphSAGE warm",        False),
    ("o1zvl1jk", "Phase 2b · DiffPool cold",        True),
    ("fp4lcpe8", "Phase 2b · DiffPool warm",        True),
]

# Parse: "Epoch  10/50  loss=1.18244  (lm=1.1592  lp=0.1171  ent=1.1544) ... val_τ=+0.1245"
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+\s+loss=([-\d.]+)")
_AUX_RE   = re.compile(r"lm=([-\d.]+)\s+lp=([-\d.]+)\s+ent=([-\d.]+)")
_TAU_RE   = re.compile(r"val_τ=([+-][\d.]+)")


def _find_log(run_id: str) -> Path:
    hits = glob.glob(str(ROOT / "wandb" / f"run-*-{run_id}" / "files" / "output.log"))
    if not hits:
        raise FileNotFoundError(f"No output.log for run {run_id}")
    return Path(hits[0])


def parse_run(run_id: str) -> Dict[str, List]:
    """Pull per-epoch loss, (optional) aux losses, and sparse val τ from a log."""
    epochs, losses = [], []
    lm, lp, ent = [], [], []
    tau_ep, tau_val = [], []
    for line in _find_log(run_id).read_text().splitlines():
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        epochs.append(ep)
        losses.append(float(m.group(2)))
        a = _AUX_RE.search(line)
        if a:
            lm.append(float(a.group(1)))
            lp.append(float(a.group(2)))
            ent.append(float(a.group(3)))
        t = _TAU_RE.search(line)
        if t:
            tau_ep.append(ep)
            tau_val.append(float(t.group(1)))
    return {"epochs": epochs, "loss": losses, "lm": lm, "lp": lp, "ent": ent,
            "tau_ep": tau_ep, "tau_val": tau_val}


def plot_training_curves() -> None:
    """val τ vs epoch for all four runs, with the paper baseline marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ["o-", "s-", "^--", "v--"]
    for (run_id, label, _), style in zip(RUNS, styles):
        d = parse_run(run_id)
        ax.plot(d["tau_ep"], d["tau_val"], style, label=label, linewidth=1.8, markersize=5)

    ax.axhline(0.34, color="crimson", linestyle=":", linewidth=1.5,
               label="TPUGraphs paper baseline (0.34)")
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Kendall's τ  (mean over 7 graphs)")
    ax.set_title("Validation ranking quality across training\n(higher = better config ranking)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def plot_loss_equilibrium(run_id: str = "o1zvl1jk") -> None:
    """The 3 competing losses for the DiffPool cold run, on twin axes.

    Tells the story of the audit: listmle + lp fall (model learns, clusters
    respect edges) while entropy *rises* (assignments stay soft, no collapse).
    """
    d = parse_run(run_id)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(d["epochs"], d["lm"], color="tab:blue",  label="ListMLE (ranking) ↓")
    ax1.plot(d["epochs"], d["lp"], color="tab:green", label="LP loss (cluster↔edge) ↓")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ListMLE / LP loss", color="black")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(d["epochs"], d["ent"], color="tab:red", label="Entropy (assignment softness) ↑")
    ax2.set_ylabel("Entropy loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # combine legends
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8.5)

    ax1.set_title("DiffPool cold start — the fragile 3-loss equilibrium\n"
                  "entropy RISES (clusters stay soft, no collapse) as ranking & LP fall")
    fig.tight_layout()
    out = FIG_DIR / "loss_equilibrium.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    plot_training_curves()
    plot_loss_equilibrium()
