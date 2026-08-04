# Phase 8c — Pre-registration (MXU-feature co-design experiment)

> **Status: FROZEN 2026-06-29, before running 8c and before the Phase-6 sweep finished.**
> This document fixes the hypotheses, the data split, the feature arms, the metric,
> the statistical test, and the decision rules **in advance**, so that no choice can be
> made after seeing results (anti-HARKing / anti-p-hacking). It is the rigor instrument
> the project is built on (see VERIFICATION.md, C10/C12). Any deviation must be recorded
> as a dated amendment below, not a silent edit.

## 1. Motivation (what we already know, with evidence)
- **C10 (Phase 5):** GNN cost models fail on convolution graphs (~47% off Top-1) vs
  transformer/UNet (~5%). Originally n=7, post-hoc — exploratory.
- **Corroboration (Phase 8 meantime analysis, `phase8_mxu_misalignment.md`):** across
  **160 xla graphs**, convolution-bearing graphs are **~2.8× more layout-sensitive**
  (real-runtime CV median 0.239 vs 0.086; ρ=+0.54), computed with **no model**. So
  convolution is a genuinely distinct, layout-sensitive regime — not a model artifact.
- **Tempering result (same analysis):** static **channel-128 padding overhead does NOT**
  predict layout sensitivity (within-conv ρ=−0.06). The hardness is the spatial-tiling /
  layout search, not channel packing. **This shapes the predictions below.**

## 2. Research question
Does **encoding hardware-grounded MXU features** of convolution shapes (Phase 8b,
`scripts/mxu_features.py`) reduce the convolution-specific Top-1 failure of the flat
cost model — and if not, does the gap **persist** as evidence of a representational
floor that static features cannot cross?

## 3. Frozen hypotheses & registered predictions
- **H1 (primary):** Adding MXU features reduces the conv-subset Top-1 slowdown vs the
  matched baseline, paired across seeds. Null H0: no change.
- **H2 (placebo/specificity):** The non-conv subset Top-1 is **unchanged** (MXU features
  are zero on non-conv nodes; any movement there signals a confound, not a fix).
- **Registered predictions (committed now, given §1 evidence):**
  - P1: the **channel-padding-only** arm will **NOT** significantly close the gap (predict null).
  - P2: the **spatial/tiling** arm (kernel area, output spatial extent, gemm_m) has the
    best — but uncertain — chance; we do **not** predict a guaranteed fix.
  - P3: a **persisting** gap is an expected, still-publishable outcome (earns the
    representational-floor claim). We commit to reporting it as a finding, not a failure.

## 4. FROZEN data split (the anti-HARKing artifact)
**Rule (objective, fixed):** a validation graph is **convolution-dominated** iff
`mxu_frac = (#conv nodes / #nodes) ≥ 0.020`; else **other**. (Threshold chosen to
reproduce the Phase-5 conv-vs-other grouping objectively; frozen here, never re-tuned.)

Applied to the xla validation graphs (identical in both collections):
- **CONV subset (3):** `inception_v3_batch_128_train`, `resnet50.4x4.fp16`,
  `resnet_v1_50_official_batch_128_bf16`
- **OTHER subset (4):** `bert_pretraining.4x4.fp16`, `mlperf_bert_batch_24_2x2`,
  `tf2_bert_pretrain_dynamic_batch_size`, `unet_3d.4x4.bf16`

Evaluation runs on **both** `xla/default/valid` and `xla/random/valid` → **6 conv-graph
eval instances** and **8 other** (same names, distinct layout-config sets). n is small and
acknowledged; it is the maximum the benchmark's held-out conv graphs allow.

## 5. Feature arms (ablation — fixed set, run all)
All arms = the SAME flat wide-baseline backbone; only the input feature dimension changes.
The MXU columns are appended to the existing 158-dim Early-Join input (158 → 158+k),
computed by `compute_mxu_features` (conv nodes only; 0 elsewhere). RAW MXU values get the
same `sign·log1p` normalisation as other features.
1. **baseline** — 158-dim (no MXU features). The control.
2. **+channel** — append c_out/k MXU-utilisation + padding-waste columns only.
3. **+spatial** — append kernel area, output spatial extent, gemm_m, n_mxu_passes.
4. **+full** — all Phase-8b MXU features.
(Benjamini–Hochberg correction across the 3 feature arms for the primary test.)

## 6. Training protocol (matched to Phase 2c/3/6 — no changes)
Wide flat baseline (`--model wide-baseline`, ~236k params), `--configs-per-batch 8`,
`--max-configs-per-graph 128`, cosine LR 1e-3→1e-5, grad-clip 0.1, dropout 0,
**5 seeds {42,100,123,456,789}**, **paired by seed**, run under the capped harness
(`launch_capped.sh`, num-workers 0). Train on the corresponding xla train split.

## 7. Primary metric & statistical test
- **Primary:** per-graph **Top-1 slowdown** on the CONV subset (best true runtime among
  the model's top-1 ÷ true optimal), as in Phase 4/5. Compare **+arm vs baseline**,
  **paired across the 5 seeds**, averaged over the conv eval instances.
- **Test:** paired t-test + Wilcoxon on the per-seed conv-subset slowdown (with vs
  without); **bootstrap 95% CI on the gap-closure**; report **effect size**, not just sign.
- **Secondary (placebo):** same on the OTHER subset — must be ~unchanged.
- **Significance threshold:** p < 0.05 after BH across arms. Given C9 (seed std ≈0.04–0.07
  dominates), an effect smaller than the seed noise is reported as null regardless of sign.

## 8. Decision rules (fixed before data)
| Outcome | Interpretation (committed) |
|---|---|
| CONV gap closes, p<0.05, OTHER unchanged | **Positive co-design result** — encoding MXU geometry fixes the conv failure. |
| CONV gap unchanged (any arm) | **Representational-floor evidence** — static MXU features insufficient; the layout-relevant info lives in the compiler tiling search. Publishable. |
| OTHER subset also moves | **Confound** — not a conv-specific fix; investigate before any claim. |
| Mixed across arms | Report per-arm; the +spatial vs +channel contrast is itself the finding (tests P1/P2). |

## 9. Anti-p-hacking commitments
- Seeds fixed at 5; **we will not add seeds** to push a result over threshold.
- The §4 split and §5 arms are **frozen**; no post-hoc regrouping, no new arms after seeing data.
- No window/cap tuning to manufacture significance (PI-endorsed discipline).
- A **persisting gap is pre-committed as a reportable finding**, removing the incentive to
  chase a positive.

---
*Amendments (date + reason) go below this line; the above is frozen.*

### Amendment A1 — 2026-07-10 (operationalizations recorded BEFORE any 8c run)
1. **Training collection clarified:** `layout:xla:random` train split — the Phase 2c/3/5
   lineage that established the conv-failure baseline. Eval on the frozen §4 split in BOTH
   xla valid sets, per §4.
2. **Exact arm columns** (from `mxu_features.FEATURE_NAMES`):
   channel = {c_out_mxu_util, c_out_pad_waste, k_mxu_util, k_pad_waste} (k=4);
   spatial = {kernel_area, gemm_m, n_mxu_passes} (k=3) — §5's "output spatial extent" is
   operationalized as gemm_m (batch × output spatial positions); full = all 16.
3. **NaN handling:** contraction features of conv nodes with unresolved kernel operands → 0.
4. **Param delta:** input-proj growth with k: baseline 236,157 / channel 236,845 / spatial
   236,673 / full 238,909 (≤ +1.2%) — inherent to the input-dim change; reported.
5. **Baseline arm is RE-RUN** under the identical capped harness (num-workers 0), not reused
   from Phase 2c (which ran with 4 workers) — strictest same-harness pairing.
6. Note: the 3-D kernel-volume fix (2026-07-07, `mxu_features.py`) predates all 8c runs.
7. **Implementation:** new files `dataset_mxu.py` / `train_8c.py` / `run_phase8c.sh`
   (dataset.py & train.py untouched); features get sign·log1p per §5; smoke-tested on all
   4 arms (dims 158/162/161/174, conv-only nonzero, finite, forward OK).

### Amendment A2 — 2026-07-20 (specificity control, registered BEFORE the run)
**Reason.** 8c landed as a POSITIVE result that *reverses* P1 (channel, predicted null, was
strongest: CONV Top-1 slowdown 1.425→1.120, BH-p.003). Because MXU features are provably
**config-invariant** (`dataset_mxu.py` computes one block per graph, appended to every config),
the effect can only run through changed training dynamics — which raises the Phase-2b question:
is it *MXU information*, or just *any extra conv-node columns / capacity*? This control decides it.

1. **New arm `scramble` (k=4, dim 162 — mirrors the strongest arm, `channel`).** Same four channel
   columns and same sign·log1p normalization, but within each graph the block's feature ROWS are
   permuted **among the MXU-resolved (any-nonzero) nodes only**, via a deterministic per-graph RNG
   (`np.random.default_rng(file_idx)`). This preserves — exactly — the value multiset, the
   conv-node placement (zeros stay on non-conv nodes), the dimensionality, and config-invariance;
   it destroys ONLY the node↔value alignment (the actual MXU information). Verified on
   `inception_v3` (293 nonzero rows: identical placement + value multiset to channel, rows
   reassigned, deterministic across reload).
2. **Registered prediction P4 (committed now, before the run):** if the 8c effect is real MXU
   co-design, **channel will significantly beat scramble** on CONV Top-1 slowdown (channel < scramble).
   If **channel ≈ scramble** (no significant gap), the 8c effect is a capacity/regularization
   artifact of adding 4 conv-localized columns, NOT MXU information → 8c must be reframed.
3. **Decision rule (fixed before data):** primary contrast = CONV-subset Top-1 slowdown, `scramble`
   vs `channel`, paired by seed (same t + Wilcoxon + bootstrap CI + BH as §7); placebo on OTHER.
   Also report scramble vs baseline (does the mere presence of the columns move anything?).
4. **Protocol:** identical to §6 / A1 — 5 seeds {42,100,123,456,789}, `layout:xla:random` train,
   eval on the frozen §4 split in both xla valid sets, same capped harness (num-workers 0).
   `channel`/`baseline` checkpoints are reused (unchanged); only `scramble` is newly trained
   (5 runs). ⛔ GPU-gated (see [[project-gpu-blocker]]).
