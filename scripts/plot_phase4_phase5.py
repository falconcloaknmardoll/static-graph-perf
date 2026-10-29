#!/usr/bin/env python3
"""
scripts/plot_phase4_phase5.py
-----------------------------
Generate the two missing figures:
  phase4_topk.png         — Top-K slowdown, flat vs hardware, K∈{1,5,10} (grouped bars)
  phase5_conv_failure.png — per-graph Top-1 slowdown, colored by network family
                            (the bimodal convolution-vs-transformer/UNet split)

Phase 5 reads the real per-graph cache (checkpoints/phase5_mispred.json).
Phase 4 uses the verified 5-seed aggregates from phase4_topk_results.md.

Run:
    python scripts/plot_phase4_phase5.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "experiments" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Network family by graph-name substring (for Phase 5 coloring)
CONV = ("resnet", "inception")  # convolution-dominated
def family(name: str) -> str:
    n = name.lower()
    if any(k in n for k in CONV):
        return "convolution"
    if "unet" in n:
        return "memory-bound (UNet)"
    return "transformer"

FAM_COLOR = {"convolution": "tab:red",
             "transformer": "tab:blue",
             "memory-bound (UNet)": "tab:green"}


def plot_phase5() -> None:
    d = json.load(open(ROOT / "checkpoints" / "phase5_mispred.json"))
    # clean display names (avoid mangling via blind string replacement)
    DISPLAY = {
        "resnet50": "resnet50", "resnet_v1_50": "resnet_v1_50",
        "inception_v3": "inception_v3", "unet_3d": "unet_3d",
        "bert_pretraining": "bert_pretraining", "mlperf_bert": "mlperf_bert",
        "tf2_bert": "tf2_bert",
    }
    def disp(name: str) -> str:
        n = name.lower()
        for key, label in DISPLAY.items():
            if key in n:
                return label
        return name.replace(".npz", "")[:22]
    rows = []
    for name, v in d.items():
        rows.append((disp(name), v["mean_top1"], family(name)))
    rows.sort(key=lambda r: r[1])  # ascending slowdown

    labels = [r[0] for r in rows]
    vals   = [r[1] for r in rows]
    cols   = [FAM_COLOR[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, alpha=0.85)
    ax.axvline(1.0, color="black", lw=1, label="optimal (1.0)")
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Top-1 slowdown vs optimal  (1.0 = perfect; lower = better)")
    ax.set_xlim(1.0, 1.62)
    ax.set_title("Where the cost models are blind — per-graph Top-1 error\n"
                 "convolution graphs ~47% off vs ~5% for transformer/UNet "
                 "(Mann-Whitney p=0.029)", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLOR.values()]
    ax.legend(handles, list(FAM_COLOR.keys()), loc="lower right", fontsize=8.5, title="network family")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG / "phase5_conv_failure.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(ROOT)}")
    # Paper version (FIG 4): same plot at column width, title removed (the caption carries it).
    ax.set_title(""); fig.set_size_inches(3.3, 2.3)
    for t in ax.texts: t.set_fontsize(6.5)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Mean Top-1 slowdown (1.0 = optimal)", fontsize=7)
    ax.legend(handles, list(FAM_COLOR.keys()), loc="lower right", fontsize=6, title="network family",
              title_fontsize=6.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        paper_out = ROOT / "paper" / "figures" / f"fig_c1_conv_failure.{ext}"
        fig.savefig(paper_out, dpi=250, **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
        print(f"wrote {paper_out.relative_to(ROOT)}")
    plt.close(fig)


def plot_phase4() -> None:
    # Verified 5-seed aggregates (phase4_topk_results.md): mean ± std, slowdown vs optimal
    K = [1, 5, 10]
    flat   = [1.214, 1.117, 1.092]
    flat_s = [0.116, 0.056, 0.043]
    hw     = [1.246, 1.084, 1.075]
    hw_s   = [0.054, 0.047, 0.038]
    pvals  = [0.65, 0.47, 0.62]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(K)); w = 0.36
    ax.bar(x - w/2, flat, w, yerr=flat_s, capsize=5, label="flat baseline", color="tab:blue", alpha=0.85)
    ax.bar(x + w/2, hw,   w, yerr=hw_s,   capsize=5, label="hardware model", color="tab:orange", alpha=0.85)
    ax.axhline(1.0, color="black", lw=1, label="optimal (1.0)")
    for xi, f, h, p in zip(x, flat, hw, pvals):
        ax.text(xi, max(f, h) + 0.16, f"p={p:.2f}\n(n.s.)", ha="center", fontsize=8.5, color="gray")
    ax.set_xticks(x); ax.set_xticklabels([f"Top-{k}" for k in K])
    ax.set_ylabel("Slowdown vs optimal  (1.0 = perfect; lower = better)")
    ax.set_ylim(1.0, 1.5)
    ax.set_title("Phase 4 — Top-K autotuner metric: no significant difference at any K\n"
                 "(both models ~21–25% off optimal at Top-1)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "phase4_topk.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    plot_phase5()
    plot_phase4()
