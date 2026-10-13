# R0 — CONV-subset Top-1 of the structured arms on xla:random (exploratory declaration)

> **Status: FROZEN 2026-10-11 (the first author sign-off, this date); nothing computed at freeze.** Freeze = the first author's sign-off,
> recorded as a dated line at the bottom, then commit, then compute. Exploratory and eval-only, so
> no pre-registration is needed. It is declared before computing, following the `6def25d`
> precedent (Top-K at tuned LRs). Origin: `experiments/framing_audit_2026-10-11.md` §3a / §4.

## 1. Why

Option A contrasts "added structure → no detectable gain" with "added MXU information → CONV gap
narrows". On xla:random at the shared lr 1e-3 the two sides already share a collection, a recipe
and a backbone. They do **not** share a metric: structure was scored on overall τ / overall Top-K,
and 8c on CONV-subset Top-1. R0 scores the structured arms on 8c's metric, so the contrast becomes
like-for-like.

## 2. What is already observed (not blind)

- **Phase 5** (`phase5_error_analysis.py:141–144`, cache `checkpoints/phase5_mispred.json`)
  already holds 5-seed-mean Top-1 per graph for **flat** (`checkpoints/ablation/wide-baseline-*`)
  and **hardware** (`checkpoints/phase3/hardware-*`) on the **xla:random** validation graphs. The
  three CONV graphs were seen while drafting this declaration (2026-10-11):
  inception_v3 flat 1.4433 / hw 1.4389; resnet50.4x4 1.511 / 1.5126; resnet_v1_50 1.4227 / 1.4803.
- **Not yet observed:** paired per-seed contrasts; the **hierarchical** arm
  (`checkpoints/ablation/diffpool-*`) on CONV; any arm on the **xla:default** copies of the 3 CONV
  graphs; the OTHER subset by arm.
- Because of the first bullet, R0 is labelled **exploratory, partially observed**. It can support
  wording; it can't be a confirmatory result.

## 3. Design (fixed now)

- **Checkpoints (existing, no training):** flat `checkpoints/ablation/wide-baseline-seed{s}`,
  hierarchical `checkpoints/ablation/diffpool-seed{s}`, hardware `checkpoints/phase3/hardware-seed{s}`,
  s ∈ {42, 100, 123, 456, 789}. All were trained on xla:random train at lr 1e-3.
- **Split:** the frozen 8c split (`phase8c_preregistration.md:45–48`). CONV = 3 graphs, OTHER = 4.
  Scored on both `xla/default/valid` and `xla/random/valid` (6 CONV / 8 OTHER instances), exactly
  as 8c.
- **Metric:** per-instance Top-1 slowdown, averaged over instances per seed, as in
  `phase8c_evaluate.py`. **Eval batch 8** (O8: the Phase 6 protocol; Phase 5 used the Phase 2c
  setting, so its numbers may differ slightly).
- **Contrasts:** hierarchical − flat and hardware − flat on CONV, paired by seed. Paired t
  (two-sided), Wilcoxon, bootstrap 95% CI, Cohen's d, realized MDE. Same on OTHER, as a placebo.
- **Reported beside, not as a test:** 8c's channel − baseline on the same metric
  (Δ −0.3043, BH-p .0032; `phase8c_results.md:12`), with the caveat that the 8c baseline and the
  Phase 2c flat are the same model and recipe under different harnesses (num_workers 0 vs 4), so
  the two flats are **not** interchangeable (0.2311 vs 0.2880 overall τ; audit §1c).

## 4. How the result will be used (fixed now)

| Outcome on CONV | Wording it supports |
|---|---|
| Both structured − flat ≥ 0, or n.s. with \|Δ\| well below 8c's 0.3043 | "On xla:random at one shared recipe, adding structure does not narrow the CONV gap; adding MXU information does." Quote the MDE. |
| A structured arm significantly < flat on CONV | Option A's xla:random contrast **does not hold** for that arm; report it, and drop "structure gives no gain" for CONV. |
| Mixed / n.s. with a large negative point estimate | "Not distinguishable at n = 5"; quote the MDE; no contrast claim. |

No outcome is dropped from the paper. Top-5/Top-10 are not substitutes if Top-1 disappoints.

## 5. Budget

Eval-only on 15 checkpoints × 14 instances. Not timed; estimated at a few hours on CPU or GPU
(8c's evaluator caches per arm under `checkpoints/phase8c/eval_cache/`). An evaluator for the
hierarchical/hardware model classes is needed (`phase8c_evaluate.py` loads only
`BaselineGraphSAGE_Large`), to be written after freeze.

## Freeze

- [x] the first author sign-off (date): 2026-10-11 (given in session; recorded by Claude)

## Amendment 1 — 2026-10-11, before anything is computed

- **Eval batch.** §3 said "eval batch 8" and also "exactly as 8c" / "as in `phase8c_evaluate.py`".
  Those conflict: 8c's CONV Top-1 was scored with `phase8c_evaluate.py` at its default batch of
  **32** (`phase8c_evaluate.py:82,161`; no `--batch-size` was passed). Eval-mode BatchNorm uses batch
  statistics (O1), so the batch can move results. Because R0 exists to match 8c's metric, the
  **primary R0 numbers use batch 32**, and batch 8 is reported beside them as a sensitivity check.
- Found while writing the evaluator, before any R0 number existed. No other change.

## Amendment 2 — 2026-10-11, operational (device), before any hardware-arm or batch-8 number exists

- The GPU run computed and cached flat and hierarchical at batch 32 (`checkpoints/r0_eval_cache/`;
  not yet read), then ran out of memory on the hardware arm at batch 32 because 8e training shares
  the GPU. The remaining cells (hardware at batch 32, every arm at batch 8) run on **CPU**.
- Inference only, same checkpoints and code. CPU-vs-GPU prediction differences are expected at the
  ~1e-5 level (the scale `verify_eval_fast.py` measured for GPU run-to-run noise), which can change
  a Top-1 pick only on near-ties. Disclosed. If the GPU frees up before R0 is written into the paper,
  hardware at batch 32 is re-scored on GPU as a check.

## Amendment 3 — 2026-10-12, operational

- The CPU run (Amendment 2) was stopped after 41 h: it had completed only part of one hardware cell
  and was holding 9 GB on a memory-starved machine while 8e trained. Its partial output was **not
  cached and is discarded.** Disclosed: its log showed per-graph hardware-arm Top-1 values (batch 32,
  xla:default) for 4 graphs (inception_v3, mlperf_bert, resnet50.4x4, resnet_v1_50), seen 2026-10-12
  while checking the process.
- **All remaining cells now run on GPU** after queues k, l and m finish (~Oct 14), so every R0
  number comes from GPU inference. Amendment 2's CPU plan is withdrawn.

## Operational note — 2026-10-13
- Amendment 3's GPU run is queued as `scripts/run_gpu_queue_2026-10-13n.sh` (waits for queues k, l, m and
  for any training process to exit; log `logs/r0_conv_top1_eval_gpu.log`).
- Incident, disclosed: at ~15:30 ET on Oct 13, `r0_conv_top1_eval.py --help` was run by mistake (the script
  takes no arguments and runs in full). It loaded the cached flat/hierarchical b32 cells without printing
  them, started the hardware arm on the shared GPU, and crashed before printing any per-graph value or
  writing a cache or results file (cache timestamps unchanged since Oct 11). No R0 number was seen.
