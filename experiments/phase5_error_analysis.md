# Phase 5 Results — Hardware Bottleneck Attribution (Error Analysis)
## TPUGraphs GNN Compiler Optimization

**Status:** complete (exploratory — 7 held-out graphs).
**Headline (hypothesis, not finding):** the cost models fail **bimodally** — near-optimal
on matmul/memory-bound graphs (~5% off Top-1) but **~47% off on convolution-dominated
graphs**. Suggests MXU convolution layout is harder for static features to capture.
**This is a lead for the test set, not a confirmed result** (n=7, post-hoc grouping).

> Code: `phase5_error_analysis.py`. Reuses verified `analyze_topology.compute_spectral_gap`
> (λ₁), `analyze_opcodes.compute_config_spread`, corrected `dataset.MXU_OPS/MEM_OPS`, and
> Phase 4's prediction pipeline. Read with [`../VERIFICATION.md`](../VERIFICATION.md) C10.

---

## 0. The question

Phases 2c/3/4 established that no architecture beats the flat baseline (all nulls), and
that even the best models leave **~21–25% on the table at Top-1** (Phase 4). Phase 5 asks
*where* that error concentrates: are the worst-mispredicted graphs structurally special?

Misprediction metric = per-graph **Top-1 slowdown** (best runtime among the model's top-1
pick ÷ true-optimal runtime), averaged over the **5 seeds** and **both models**, on the
**7 held-out validation graphs**.

---

## 1. The result — a clean bimodal split by architecture family

| Graph | family | mean Top-1 slowdown |
|---|---|---:|
| unet_3d | memory-bound | 1.010 |
| tf2_bert_pretrain | transformer | 1.056 |
| bert_pretraining | transformer | 1.067 |
| mlperf_bert | transformer | 1.072 |
| inception_v3 | **convolution** | **1.441** |
| resnet_v1_50 | **convolution** | **1.452** |
| resnet50 | **convolution** | **1.512** |

**All three convolution graphs are worse than all four non-convolution graphs — perfect
separation.** Convolution graphs sit at ~47% off-optimal; transformer/UNet graphs at ~5%.

- Group test (Mann-Whitney, CNN > others, one-tailed): U=12, **p = 0.029** — which is the
  *floor* p-value achievable for a 3-vs-4 split (i.e. "as significant as 7 graphs allow").
- Top structural correlate across all 7 graphs: **`mxu_frac`** (Spearman ρ = **+0.68**,
  p = 0.09). Convolution is opcode 26 ∈ MXU_OPS, so this literally reads "more convolutions
  → worse Top-1." Second: `diameter` (ρ = −0.69, p = 0.085).

### Structural feature table (dataset norm vs worst-mispredicted)

| feature | dataset (25 train) | all val | worst (resnet50) | z |
|---|---:|---:|---:|---:|
| n_nodes | 11741 | 12797 | 5673 | −0.7 |
| mxu_frac | 0.0250 | 0.0220 | 0.0284 | +0.3 |
| mem_frac | 0.1252 | 0.1078 | 0.1049 | −0.3 |
| compute_ratio | 0.592 | 0.634 | 0.602 | +0.1 |
| norm_config_spread | 0.336 | 0.365 | 0.400 | +0.9 |
| spectral_gap λ₁ | 0.0033 | 0.0024 | 0.0064 | +0.8 |
| diameter | 20 | 17 | 15 | −0.6 |

(The single-graph "worst 10%" column z-scores are not interpretable — see §3. The
*grouped* CNN-vs-other comparison in §1 is the meaningful view.)

---

## 2. The coherent hardware story (hypothesis)

The models are **near-perfect on transformer graphs** (dominated by `dot`/matmul, whose
layout space is relatively well-behaved) and on **UNet** (memory-bound, runtime mostly
bandwidth-limited and insensitive to the layout knobs). They **collapse on convolution
graphs**. A plausible mechanism: **convolution tiling/layout on the systolic MXU has many
more performance-relevant degrees of freedom** (spatial dims, channel blocking, im2col vs
direct) than a matmul, and the static node features (shape, opcode, a few config scalars)
do not encode enough to rank those layouts. The model is effectively **blind to the part
of the search space that matters most for convolutions.**

---

## 3. ⚠️ Epistemic status — why this is a hypothesis, not a finding

This is the project's most interesting signal, which is exactly why it needs the harshest
scrutiny. Three reasons it is **not** yet a result:

1. **Post-hoc grouping (HARKing risk).** The CNN-vs-transformer line was drawn *after*
   seeing which graphs were worst. Perfect 3-vs-4 separation has a floor p of 0.029; the
   "significance" is partly manufactured by choosing the grouping that maximizes it.
2. **n = 7, no multiple-comparison control.** Nine features were tested. The two
   correlates (`mxu_frac` +0.68, `diameter` −0.69) are p ≈ 0.09 — **not** significant; at
   n=7 you need |ρ| ≳ 0.78. The "worst 10%" is literally **one graph** (resnet50), so its
   z-scores are noise (consistent with VERIFICATION C9).
3. **"Non-deterministic" is unproven.** The data show static features fail to *rank*
   convolution configs — equally consistent with the conv layout→runtime map being
   *complex but deterministic* (hard) vs *noisy* (non-deterministic). **Structure cannot
   establish non-determinism.** That needs a direct measurement (runtime variance among
   near-identical configs), which we did not compute.

---

## 4. The honest claim, and the next experiment

**Defensible wording for the paper:**

> "Learned static cost models exhibit a bimodal failure mode on TPUGraphs layout ranking:
> near-optimal on matmul-bound (transformer) and memory-bound (UNet) graphs (~5% off at
> Top-1), but ~47% off on convolution-dominated graphs (ResNet, Inception). We hypothesize
> that MXU convolution tiling is more layout-sensitive than static graph features capture.
> With 7 held-out graphs and a post-hoc grouping this is exploratory (Mann-Whitney
> p = 0.029, the 3-vs-4 floor); confirmation requires the official test set and a direct
> config-runtime-variance measure to separate 'complex' from 'non-deterministic'."

**Concrete next experiments to promote this to a finding:**
1. Run the same per-graph Top-1 analysis on the **official/larger test set** (more conv vs
   non-conv graphs) — pre-register the conv-vs-other split *before* looking.
2. Compute a **runtime non-determinism measure** per graph (variance of runtime among
   configs that differ only trivially) and test whether conv graphs are noisier — this is
   what would actually justify the word "non-deterministic."
3. Within conv graphs, correlate Top-1 error with **per-convolution layout-space size**.

---

## 5. Files

| File | Role |
|---|---|
| `phase5_error_analysis.py` | misprediction + structural features + correlations + honesty guard |
| `checkpoints/phase5_mispred.json` | cached per-graph Top-1/5/10 slowdowns (5-seed mean) |
| `analyze_topology.py` / `analyze_opcodes.py` | reused verified feature computations |

---

## 6. Bottom line

> The cost models fail bimodally: ~5% off-optimal on matmul/memory-bound graphs, ~47% off
> on convolution graphs (perfect 3-vs-4 separation, p=0.029 floor). The leading hypothesis
> — convolution MXU layout is under-captured by static features — is the project's most
> promising lead, but with n=7 and a post-hoc grouping it is **exploratory**. It hands the
> paper a concrete, falsifiable next experiment rather than a finished claim.
