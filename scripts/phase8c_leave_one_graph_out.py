"""Post-hoc robustness of the registered 8c channel effect: drop each CONV graph in turn.

Reads experiments/phase8c_results.json (raw per-instance Top-1, batch 32); no inference.
Exploratory (not pre-registered). Writes experiments/phase8c_leave_one_graph_out.md.
  python scripts/phase8c_leave_one_graph_out.py
"""
import json
import numpy as np
from scipy import stats

raw = json.load(open("experiments/phase8c_results.json"))["raw"]
CONV = ["inception_v3_batch_128_train", "resnet50.4x4.fp16", "resnet_v1_50_official_batch_128_bf16"]
SEEDS, COLLS = ["42", "100", "123", "456", "789"], ["xla/default", "xla/random"]


def per_seed(arm, graphs):
    return np.array([np.mean([raw[arm][f"{c}::{g}"][s] for c in COLLS for g in graphs]) for s in SEEDS])


lines = ["# 8c — leave-one-graph-out (post hoc, exploratory)", "",
         "Channel − baseline CONV Top-1 (batch 32), paired by seed, dropping one CONV graph (both "
         "collections) at a time. Source: `phase8c_results.json` raw. Not pre-registered.", "",
         "| dropped | baseline | channel | Δ | t-p | seeds improving |", "|---|---|---|---|---|---|"]
for drop in [None] + CONV:
    gs = [g for g in CONV if g != drop]
    b, ch = per_seed("baseline", gs), per_seed("channel", gs)
    d = ch - b
    lines.append(f"| {drop or 'none (registered)'} | {b.mean():.4f} | {ch.mean():.4f} | {d.mean():+.4f} | "
                 f"{stats.ttest_rel(ch, b).pvalue:.4f} | {(d < 0).sum()}/5 |")
lines += ["", "Per graph (mean over both collections and 5 seeds):", "", "| graph | baseline | channel |", "|---|---|---|"]
for g in CONV:
    lines.append(f"| {g} | {per_seed('baseline', [g]).mean():.4f} | {per_seed('channel', [g]).mean():.4f} |")
lines += ["", "Caveat: two of the three graphs are ResNet-50 variants, so this is two architecture "
          "families. Graph-level inference stays weak; R4 (`r4_heldout_conv_declaration.md`) tests held-out families."]
open("experiments/phase8c_leave_one_graph_out.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
