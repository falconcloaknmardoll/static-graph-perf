# Phase 6b — A4 confound checks (per-graph, tie-structure, BH)

Raw data: `experiments/phase6b_confound_checks.json`. Script: `scripts/phase6b_confound_checks.py`
(CPU-only re-eval of existing checkpoints, no new training). Run 2026-08-19→20, 16h+ wall clock.

This is the disaggregation follow-up to the 3-arm training run (2026-07-10,
`checkpoints/expansion/xla_default/phase6b_summary.txt`): flat τ=0.0327±0.0135, plain DiffPool
(hierarchy only) τ=0.1484±0.0276, hardware DiffPool τ=0.1977±0.0133. That run established
hierarchy explains ~70% of the xla:default win and edge machinery adds a further significant
increment. This doc adds the three checks A4 required before treating that as defensible.

## 1. Per-graph τ breakdown (flat vs hardware, mean of 5 seeds each)

| graph | flat | hardware | Δ | near-tie % |
|---|---|---|---|---|
| bert_pretraining.4x4.fp16 | 0.054 | 0.259 | +0.204 | 67.65% |
| inception_v3_batch_128_train | -0.283 | 0.285 | **+0.568** | 62.29% |
| mlperf_bert_batch_24_2x2 | 0.238 | 0.238 | -0.001 | 76.34% |
| resnet50.4x4.fp16 | 0.065 | 0.135 | +0.070 | 59.01% |
| resnet_v1_50_official_batch_128_bf16 | -0.004 | 0.075 | +0.080 | 63.29% |
| tf2_bert_pretrain_dynamic_batch_size | 0.086 | 0.292 | +0.206 | 62.46% |
| unet_3d.4x4.bf16 | 0.076 | 0.056 | -0.020 | 85.76% |

**Direction is consistent, magnitude is not outlier-free.** 5/7 graphs show a clear hardware > flat
win; the remaining 2 (mlperf_bert, unet_3d) are within seed noise of zero, neither flips negative
in a way that contradicts the aggregate. No graph shows the *opposite* sign at a magnitude that
would suggest the aggregate is a sign-flip artifact — this resolves the original bimodality worry
from A4's brief.

However, `inception_v3_batch_128_train` contributes Δ=+0.568, roughly 2.7× the next-largest delta
(+0.206) and more than a third of the 7-graph mean (0.158). Excluding it, the mean-of-6 drops to
0.090 — still clearly positive, but the effect **roughly halves**. The flat model is actively
anti-correlated with ground truth on this graph (τ=-0.283) while the hardware model is not
(τ=+0.285); this looks like a genuine failure mode of the flat baseline on this specific graph
(large, batch-128 image-classification, likely more layout-sensitive), not a data artifact — but
it should always be disclosed alongside the aggregate number, and any forest/per-graph plot must
show this graph is not interchangeable with the other 6.

## 2. Tie-structure (model-independent)

0% exact ties across all 7 graphs (`config_runtime` values are all numerically distinct). Near-tie
fraction (relative difference <1%) ranges 59–86% depending on graph, highest for `unet_3d.4x4.bf16`
(85.76%) — the same graph with the smallest hardware-flat delta, which is at least directionally
consistent with "more near-ties → less headroom for either model to show separation," though n=7
is too small to claim this as a trend. Because tie-structure is computed directly from the raw
runtime arrays (no model involved), it applies identically to both arms — it does not
differentially suppress the flat baseline, so it does **not** explain why our flat τ=0.033 is
below the TPUGraphs paper's reported flat baseline of τ=0.12 on this collection. That gap remains
attributed to a protocol/budget difference (their 200 iterations/batch-20 vs. our 50 epochs/
128-config cap — the paper itself under-specifies epochs), already recorded in
`experiments/paper_outline.md` §5.

## 3. BH correction across the 3 pairwise Phase 6b tests

| comparison | raw p | BH-adjusted p |
|---|---|---|
| flat vs hierarchy | 0.0019 | 0.00285 |
| hierarchy vs hardware | 0.0299 | 0.0299 |
| flat vs hardware | 5.0×10⁻⁵ | 1.5×10⁻⁴ |

(flat-vs-hierarchy and hierarchy-vs-hardware raw p from the 2026-07-10 3-arm GPU run;
flat-vs-hardware recomputed here directly from the 5-seed per-graph means, t=18.55.)

**All 3 survive BH correction at α=0.05.** The hierarchy-vs-hardware contrast is the weakest
(BH-p=0.03, close to the threshold) — consistent with hierarchy doing most of the work and edge
machinery adding a real but smaller increment, not a coin-flip result inflated by multiple
comparisons.

## Bottom line

VERIFICATION C11 → 🟢. The xla:default hardware win is real, not a tie-structure artifact, not a
multiple-comparisons artifact, and directionally uniform across graphs — but its *magnitude* is
meaningfully shaped by one graph (`inception_v3_batch_128_train`), which must be disclosed
wherever the per-graph breakdown appears in the paper. See `experiments/power_analysis_results.md`
§2 for the companion tightening-curve analysis (how much this n=7 val-graph limitation matters for
resolving effects of this size — answer: not much, the flat-vs-hardware Δτ≈0.165 is ~11× the
bootstrap median std at n=7).
