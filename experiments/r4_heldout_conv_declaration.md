# R4 — Does the 8c channel effect generalise to held-out convolution families? (exploratory)

> **Status: FROZEN ON COMMIT, 2026-10-11.** the first author approved drafting and queueing R4 in session
> (2026-10-11). He has not yet reviewed the design line by line. R4 cannot start before 8e's grid
> (and any 8e boundary step) finishes (~Oct 15), so **any change is a dated amendment made before the
> first R4 run starts.** Exploratory: designed after the 8c results were known. It never replaces 8c.
> Origin: `framing_audit_2026-10-11.md` §4 (R4) and the 3-graph objection (DEFENSIBILITY O20).

## 0. Observation status

- **Known:** all 8c / 8c-control / 8d results on the frozen validation split, and the post-hoc
  leave-one-graph-out check (`phase8c_leave_one_graph_out.md`).
- **Never computed:** any model prediction or Top-1 on the held-out graphs below. Their selection
  used only names and the static `mxu_frac` (conv nodes ÷ nodes). No runtime was read.

## 1. Question

8c's CONV subset is 3 validation graphs from two architecture families (Inception; ResNet-50, twice).
Does adding the `channel` MXU columns also improve Top-1 on **convolution families the model never
trained on and that are absent from 8c's test set**?

## 2. Held-out set (fixed now, by rule)

- **Pool:** xla:random **train** graphs that are conv-dominated by 8c's frozen rule
  (`mxu_frac ≥ 0.020`, `phase8c_preregistration.md:40–42`): 33 of 69.
- **Rule:** hold out **every conv-dominated family that is not Inception or ResNet**, i.e. not a
  family in 8c's test set. A family is the architecture named before batch/shard/precision/topology
  tokens. Held out (11 graphs, 6 families):
  - alexnet: `alexnet_train_batch_32`
  - Mask R-CNN: `mlperf_maskrcnn_1_shard_batch_4`, `mlperf_maskrcnn_batch_2`, `mlperf_maskrcnn_batch_4`
  - SSD: `mlperf_ssd_1_shard_batch_8_fast_epoch`, `mlperf_ssd_2_shard_batch_8_fast_epoch`
  - MnasNet: `mnasnet_a1_batch_128`
  - RetinaNet: `retinanet.2x2.fp32`, `retinanet.4x4.bf16.performance`, `retinanet.4x4.fp32`
  - Xception: `xception_imagenet`
- **Kept for training:** the other 58 xla:random train graphs (22 conv-dominated, all Inception/ResNet
  variants, plus every non-conv graph).
- **Disclosed limits:** Mask R-CNN, SSD and RetinaNet use ResNet-style backbones, so "new family"
  is about the whole graph, not every layer. The split is non-random by design. It targets
  the two-family objection.

## 3. Design

- **Arms:** `baseline` and `channel`, exactly 8c's pipeline (`train_8c.py`, `dataset_mxu.py`).
- **Recipe:** 8c's registered recipe (**LR changed by Amendment 1: each arm's 8e LR\***), 50 epochs, capped harness, GPU, seeds
  {42, 100, 123, 456, 789}, arms interleaved within seed. (Amendment 1 replaced the original "lr 1e-3, as registered in 8c", which would have inherited O18.)
- **Training data:** `layout/xla/random_r4/train` = symlinks to the 58 kept graphs.
  `layout/xla/random_r4/valid` = symlinks to the unchanged xla:random validation graphs, so
  checkpoint selection (overall val τ, batch 8) is exactly as in 8c. Built by `scripts/make_r4_split.py`.
- **Evaluation instances:** each held-out graph's xla:random train-split file (11) plus its
  xla:default train-split copy where one exists (8: all but the three Mask R-CNN graphs) = **19
  instances**. Those xla:default files were never in any R4 training set (R4 trains on xla:random only).
- **Metric:** Top-1 slowdown per instance (8c's definition), averaged per seed over the 19
  instances. **Eval batch 32** (as 8c; O19), with batch 8 as sensitivity.
- **Primary (exploratory):** paired Δ = held-out CONV Top-1 (channel − baseline), two-sided paired t,
  n = 5. Reported with Wilcoxon (floor .0625), bootstrap 95% CI, d, realized MDE, per-family and
  per-instance tables, and a leave-one-family-out check.
- **Placebo:** 8c's frozen OTHER validation subset (4 graphs × 2 collections), channel − baseline.

## 4. How it will be used (fixed now)

| Outcome | Wording it supports |
|---|---|
| Δ < 0, p < .05 | "In an exploratory test on six held-out convolution families, the effect generalises." Effect size and MDE quoted. |
| Δ < 0, p ≥ .05 | "Direction consistent on held-out families, not significant at n = 5 (MDE …)." No "trending". |
| Δ ≥ 0 | "The effect does not generalise beyond the Inception/ResNet families tested in 8c." Scope §7 to those families. |

Reported whichever way it falls. **Cutoff:** all 10 runs must finish by 2026-10-23 23:59 ET,
otherwise R4 is not reported. No partial result, and no seeds added.

## 5. Budget and order

10 runs × ≈ 3 h ≈ 30 GPU-h. Queue `scripts/run_gpu_queue_2026-10-11l.sh` waits for 8e's grid
(queue k), runs any 8e boundary step first (8e takes priority), then R4.

## Amendment 1 — 2026-10-11, before any R4 run exists (external review)

- **LR.** §3 trained both arms at lr 1e-3, the untuned LR that O18 is about. If 8e finds an arm's
  best LR is not 1e-3, a lr-1e-3 R4 would inherit exactly the objection 8e answers. **Changed:
  each arm trains at its final 8e LR\*** (8e §4 rule, after any boundary step), computed
  mechanically by `scripts/phase8e_lr_star.py --print-lr-star` from 8e's selection metric (overall
  val τ), never from any R4 or CONV Top-1 number. If both LR\* are 1e-3, nothing else changes. If
  either LR\* cannot be determined (8e grid incomplete), **R4 does not run.**
- Consequence: R4 now tests generalisation of the *tuned* contrast (8e Q1's arms), so it is read
  alongside 8e Q1, not 8c's lr-1e-3 contrast. The §4 table is unchanged.
- No other change. Checkpoints go to `checkpoints/r4/xla_random_r4/mxu-<arm>-lr<LR*>-seed<s>`.
