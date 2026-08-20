# Phase 6c — Cheap mechanistic checks on the xla:default hierarchy effect (2026-08-17)

> Read-only / inference-only reconnaissance. No new training. Answers the "why does hierarchy +
> hardware edges help specifically on xla:default" question left as an untested hypothesis in
> `experiments/phase6_interim_xla_default.md` and the TODO in `experiments/paper_outline.md` §5.
> Figures: `experiments/figures/config_subtlety_comparison.png`,
> `experiments/figures/diffpool_S_comparison_multi.png`.

## 1. Is xla:default's config space actually measurably subtler? — YES, 7/7 graphs

For every graph present in both collections, computed straight from `node_config_feat` in the
`.npz` files (no model involved): RMS pairwise distance between sampled configs (Euclidean, via
the variance identity `E[||xi-xj||²] = 2·Σvar`, so no O(n²) blowup even at c≈19k configs), and
mean per-node Shannon entropy of each configurable node's 18-dim layout choice across configs.

| graph | RMS pairdist (default) | RMS pairdist (random) | entropy bits (default) | entropy bits (random) |
|---|---:|---:|---:|---:|
| bert_pretraining | 45.91 | 51.08 | 1.300 | 1.532 |
| inception_v3 | 62.05 | 75.01 | 2.088 | 2.978 |
| mlperf_bert | 96.13 | 105.24 | 1.270 | 1.816 |
| resnet50 | 45.52 | 55.05 | 2.104 | 2.942 |
| resnet_v1_50 | 46.74 | 54.94 | 2.222 | 2.942 |
| tf2_bert_pretrain | 47.35 | 51.71 | 1.302 | 1.507 |
| unet_3d | 32.12 | 34.86 | 1.746 | 2.226 |

**xla:default is lower on both metrics for all 7/7 graphs, no exceptions.** Magnitude varies
(8–17% smaller spread, 14–30% less entropy depending on the graph) but the direction is perfectly
consistent. This turns "xla:default configs are subtle and clustered" from a plausible-sounding
hand-wave into an actual number — real, clean, worth citing as-is.

**One data-quality note surfaced while building this:** config index 0 is a coincidental shared
baseline row — `node_config_feat[0]` is bit-identical between the default and random files for
inception_v3 (and index 0 also equals index 1 within the same file). This looks like a fixed
reference/no-override config both collections include, not a bug — but it means **don't sample
config index 0 as "representative"** in any future check; anything downstream of this file should
pick an arbitrary interior index instead.

## 2. Does DiffPool's layer-1 clustering (matrix S) actually look different by collection? — NO, not for a single config

Extracted `S` from `HierarchicalGraphSAGE.pool1` (checkpoint `checkpoints/expansion/xla_default/
hierarchical-seed42/best_model.pt`, epoch 45, val_τ=0.174) for one non-degenerate config (index 50)
on 3 representative graphs, comparing the same graph under xla:default vs xla:random:

| graph | collection | clusters used (of K=128) | cluster-size entropy | mean assignment peakedness | cluster sizes (top 4) |
|---|---|---:|---:|---:|---|
| inception_v3 (N=12067) | default | 4 | 1.907 b | 0.5463 | 3843, 3783, 3096, 1345 |
| inception_v3 | random | 4 | 1.906 b | 0.5485 | 3897, 3775, 3048, 1347 |
| resnet50 (N=5673) | default | 4 | 1.944 b | 0.5423 | 1924, 1540, 1361, 848 |
| resnet50 | random | 4 | 1.944 b | 0.5422 | 1921, 1540, 1362, 850 |
| bert_pretraining (N=21335) | default | 4 | 1.692 b | 0.4250 | 11516, 4207, 3713, 1899 |
| bert_pretraining | random | 4 | 1.691 b | 0.4255 | 11516, 4310, 3596, 1913 |

**These are essentially the same distribution across collections, on every graph checked.**
Cluster count, entropy, peakedness, and even the individual cluster sizes match within noise.
Plot: `experiments/figures/diffpool_S_comparison_multi.png`.

## Honest interpretation

The two checks point in different directions and that's the real finding: the config *space* is
measurably subtler on xla:default (§1), but a single config's pooling *structure* doesn't visibly
reflect that (§2) — clustering is dominated by the static graph (node_feat + topology), and the
18-dim config perturbation isn't enough to move which nodes land in which cluster. **The simple
version of the mechanistic story — "DiffPool visibly clusters xla:default differently because
its configs are subtler" — is not supported by this check.** That doesn't kill the hierarchy
result (Phase 6/6b's τ gain is real and already established independently), it just means the
*explanation* for why hierarchy helps specifically on xla:default isn't "the clusters look
different," at least not at the single-snapshot level tested here.

**If this mechanism question matters enough to pursue further** (e.g. The second author or a reviewer pushes on
it), the next cheapest step is NOT more single-snapshot extraction — it's measuring **how much S
(or the final score) shifts when the config changes**, within one collection: if hierarchy's
advantage is about resolving subtle configs, xla:default's tightly-clustered configs should still
need to produce *distinguishable* downstream scores for the model to rank them correctly, so the
informative quantity is the *sensitivity* of S/score to a config perturbation, not S's shape for
any single config. Still no new training required (same checkpoints, just more forward passes
across multiple config indices per graph) — flagged as a candidate for later, not built now.
