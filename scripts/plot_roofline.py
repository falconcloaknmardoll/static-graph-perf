#!/usr/bin/env python3
"""
Phase 8 — static roofline characterization of convolution nodes, overlaid with
where the learned models fail (Phase-5 per-graph Top-1 slowdown).

WHAT THIS FIGURE IS (honest scope):
  A *static* roofline-style plot — per conv node, arithmetic intensity (MACs per
  element accessed; for 2-byte dtypes this ≈ FLOPs/byte) vs MXU systolic-array
  packing utilization (c_out_util × k_util), MAC-weighted. It characterizes the
  WORKLOAD, not measured performance (no FLOP/s axis — we have no per-node
  timings). The overlay is the graph-level model failure from Phase 5.

WHY (identity + science):
  The architect-coded question: do the convolutions that break the learned
  performance model live in a specific roofline regime (compute-bound, deep in
  MXU territory) vs the benign ones (UNet's memory-bound 3D convs)? If they
  separate, the conv failure is attributable to a *hardware regime*, not just
  an op type.

  Ridge-point annotation uses TPU v3 = 123 TFLOP/s (bf16) peak per chip and
  900 GB/s HBM per chip → ridge = 123e12/900e9 ≈ 136.7 ≈ 137 FLOP/byte. VERIFIED
  2026-08-25 against Google's official Cloud TPU v3 spec page (docs.cloud.google.com/
  tpu/docs/v3: "Peak compute per chip: 123 teraflops (bf16)"; "HBM2 capacity and
  bandwidth: 32 GiB, 900 GBps"), the architecture Jouppi et al. 2020 describes.
  Note: `papers/jouppi2020_10_tpu-v2-v3.pdf` in this repo is a Hot Chips 2020
  slide deck, not the full ACM paper — it does not itself contain a numeric
  specs table, so the 123/900 figures are confirmed via the vendor spec page,
  not extracted from that PDF. For bf16 (2 B/elem): FLOPs/byte = 2·MACs /
  (2·elems) = MACs/elem, so the ridge lands at ~137 on our x-axis for bf16
  graphs.

Output: experiments/figures/phase8_roofline.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxu_features import compute_mxu_features, CONV_OPCODE  # noqa: E402

VALID = "data/predict-ai-model-runtime/npz_all/npz/layout/xla/default/valid"
# The 4 conv-bearing Phase-5 val graphs: (file-stem, label, Top-1 slowdown, color)
GRAPHS = [
    ("unet_3d.4x4.bf16",                     "UNet-3D (mem-bound)",  1.010, "#2c7fb8"),
    ("inception_v3_batch_128_train",         "Inception-v3",         1.441, "#fd8d3c"),
    ("resnet_v1_50_official_batch_128_bf16", "ResNet-v1-50",         1.452, "#e34a33"),
    ("resnet50.4x4.fp16",                    "ResNet-50",            1.512, "#b30000"),
]
RIDGE_BF16 = 137.0  # MACs/elem ≈ FLOP/byte @ bf16 — VERIFIED 2026-08-25 (123 TFLOP/s ÷ 900 GB/s, TPU v3/chip)


def load_conv_points(stem: str):
    z = np.load(f"{VALID}/{stem}.npz", mmap_mode="r")
    op = np.asarray(z["node_opcode"])
    feats, names = compute_mxu_features(
        np.asarray(z["node_feat"]), op, np.asarray(z["edge_index"]))
    ci = {n: i for i, n in enumerate(names)}
    conv = np.where(op == CONV_OPCODE)[0]
    ai = feats[conv, ci["arithmetic_intensity"]].astype(float)
    util = (feats[conv, ci["c_out_mxu_util"]] * feats[conv, ci["k_mxu_util"]]).astype(float)
    macs = np.power(10.0, feats[conv, ci["log10_macs"]].astype(float))
    ok = np.isfinite(ai) & np.isfinite(util) & np.isfinite(macs)
    return ai[ok], util[ok], macs[ok]


def paper_figure(summary) -> None:
    """Appendix version: page width, no titles, legend clear of the data."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.7, 1.0]})
    for (stem, _, _, _), (label, slow, ai_w, util_w, n, color) in zip(GRAPHS, summary):
        ai, util, macs = load_conv_points(stem)
        size = 4 + 34 * (np.log10(macs) - 7.5).clip(0, 3) / 3
        axA.scatter(ai, util, s=size, alpha=0.6, color=color, edgecolors="white", linewidths=0.25,
                    label=f"{label.replace(' (mem-bound)', '')} ({slow:.2f})")
    axA.axvline(RIDGE_BF16, color="#6e7781", ls=(0, (2, 2)), lw=0.6)
    axA.text(RIDGE_BF16 * 1.15, 0.05, "TPU v3 ridge ≈137 FLOP/byte", fontsize=7, color="#6e7781")
    axA.set_xscale("log")
    axA.set_xlabel("Arithmetic intensity (FLOP/byte at bf16)")
    axA.set_ylabel("MXU packing utilisation")
    axA.set_ylim(0, 1.05)
    axA.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=7,
               handletextpad=0.1, columnspacing=0.8, borderpad=0, borderaxespad=0.3)
    ys = np.arange(len(summary))[::-1]
    for y, (label, slow, ai_w, util_w, n, color) in zip(ys, summary):
        axB.barh(y, slow - 1.0, left=1.0, color=color, height=0.68)
        axB.text(slow + 0.012, y, f"{slow:.2f}", fontsize=7, va="center", color="#1f2328")
    axB.set_yticks(ys)
    axB.set_yticklabels([f"{l.replace(' (mem-bound)', '')}\nintensity {a:,.0f}, util {u:.2f}"
                         for l, _, a, u, _, _ in summary], fontsize=7)
    axB.tick_params(axis="y", length=0)
    axB.set_xlim(1.0, 1.62)
    axB.set_xticks([1.0, 1.2, 1.4, 1.6]); axB.set_xlabel("Top-1 slowdown")
    for ax in (axA, axB):
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig("paper/figures/appendix_roofline.png", dpi=300)
    plt.close(fig)
    print("wrote paper/figures/appendix_roofline.png")


def main() -> int:
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1.0]})

    summary = []
    for stem, label, slow, color in GRAPHS:
        ai, util, macs = load_conv_points(stem)
        w = macs / macs.sum()
        ai_w = float(np.sum(w * ai))
        util_w = float(np.sum(w * util))
        summary.append((label, slow, ai_w, util_w, len(ai), color))
        size = 18 + 140 * (np.log10(macs) - 7.5).clip(0, 3) / 3
        axA.scatter(ai, util, s=size, alpha=0.55, color=color,
                    edgecolors="k", linewidths=0.3,
                    label=f"{label} — slowdown {slow:.2f}")

    axA.axvline(RIDGE_BF16, color="grey", ls="--", lw=1)
    axA.text(RIDGE_BF16 * 1.15, 0.32,
             "TPU v3 ridge ≈137 FLOP/B (bf16)\n← memory-bound | compute-bound →",
             fontsize=7.5, color="dimgrey")
    axA.set_xscale("log")
    axA.set_xlabel("Arithmetic intensity  (MACs per element accessed ≈ FLOP/byte @ bf16)")
    axA.set_ylabel("MXU systolic packing utilization  (c_out util × K util)")
    axA.set_ylim(0, 1.05)
    axA.set_title("Static roofline position of convolution ops\n"
                  "(size ∝ MACs; color = graph, with its model failure)")
    axA.legend(loc="lower left", fontsize=7.5, framealpha=0.92)
    axA.grid(alpha=0.25)

    # Panel B: the point — NO static axis orders the failure. Horizontal bars of
    # model failure, annotated with each graph's static MXU character.
    ys = np.arange(len(summary))[::-1]
    for y, (label, slow, ai_w, util_w, n, color) in zip(ys, summary):
        axB.barh(y, slow - 1.0, left=1.0, color=color, alpha=0.85,
                 edgecolor="k", linewidth=0.4, height=0.62)
        axB.text(1.005, y + 0.36, f"{label}", fontsize=8, fontweight="bold")
        axB.text(max(slow + 0.012, 1.02), y - 0.06,
                 f"AI≈{ai_w:,.0f}, util {util_w:.2f}",
                 fontsize=7.3, color="dimgrey", va="center")
    axB.set_yticks([])
    axB.set_xlim(1.0, 1.62)
    axB.set_xlabel("Phase-5 Top-1 slowdown (1.0 = optimal)")
    axB.set_title("Model failure vs static MXU character:\n"
                  "neither AI nor packing utilization orders it")
    axB.grid(alpha=0.25, axis="x")

    fig.suptitle("Where the learned TPU performance model fails, on the roofline plane",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out = Path("experiments/figures/phase8_roofline.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")

    plt.close(fig)
    paper_figure(summary)

    print(f"\n{'graph':<24}{'slowdown':>9}{'MAC-wtd AI':>12}{'MAC-wtd util':>13}{'#conv':>7}")
    for label, slow, ai_w, util_w, n, _ in summary:
        print(f"{label:<24}{slow:>9.3f}{ai_w:>12.1f}{util_w:>13.2f}{n:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
