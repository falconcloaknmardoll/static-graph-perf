# O19 — 8c/8d CONV Top-1 at eval batch 8 (exploratory sensitivity, eval-only)

> **Status: FROZEN ON COMMIT, 2026-10-13**, before any batch-8 number exists. The first author approved it in
> session (2026-10-13). It runs after R0 (queue `scripts/run_gpu_queue_2026-10-13o.sh`
> waits for queues k, l, m and n). Any change after freezing is a dated amendment made before
> the queue starts. Origin: DEFENSIBILITY O19 (`DEFENSIBILITY.md:69`). It is grouped with O1/O8
> as a single eval-batch item in the paper.

## 0. Observation status

- **Known (registered, batch 32):**
  - 8c primary CONV Δ vs baseline: channel −0.3043 (BH-p .0032), spatial −0.2488 (.0014), full
    −0.2313 (.0073).
  - OTHER placebo: channel +0.0020 (p .82), spatial −0.0035 (.46), full −0.0147 (.020)
    (`phase8c_results.md`).
  - P4 channel − scramble −0.2442 (p .0063), with the scramble OTHER placebo +0.0079 (p .028)
    (`phase8c_control_results.md:29-30`).
  - 8d oracle − channel −0.0725 (p .19), oracle CONV 1.0478, recovery 0.8077 (`phase8d_results.md`).
- **Known, about this check:** the analysis code was dry-run on the batch-32 caches only. It
  reproduces every number above exactly. No inference was run.
- **Never computed:** any 8c or 8d Top-1 at a batch other than 32.
- **Already at batch 8:** checkpoint selection. All 8c/8d checkpoints were selected on
  training-time val τ at batch 8 (saved `args.eval_batch = 8`, e.g.
  `checkpoints/phase8c/xla_random/mxu-channel-seed42/best_model.pt`). Only the Top-1 scoring
  used 32 (`phase8c_evaluate.py:82,161`; `phase8d_evaluate.py`, `--batch-size` default 32).

## 1. Question

Do 8c's registered verdicts (the primary arms, the placebo, P4) and 8d's descriptives hold when
CONV Top-1 is scored at eval batch 8, the batch the Phase 6 τ results use? BatchNorm uses batch
statistics at eval time (`model.py:86-91`, `track_running_stats=False`), so the scores can depend
on the batch.

## 2. Design (fixed now)

- **Checkpoints:** existing ones only, with no training.
  - `checkpoints/phase8c/xla_random/mxu-{baseline,channel,spatial,full,scramble}-seed<s>`
  - `checkpoints/phase8d/xla_random/oracle-bucket-seed<s>`
  - Seeds {42, 100, 123, 456, 789}.
- **Metric, split, statistics:** unchanged from `phase8c_evaluate.py` / `phase8d_evaluate.py`.
  - Top-1 slowdown = runtime[argmax] / min runtime, per graph.
  - Frozen CONV (3) / OTHER (4) split, on both xla valid sets.
  - Per-seed subset means, paired by seed.
  - Two-sided paired t, Wilcoxon, bootstrap CI and MDE.
  - BH over the three 8c primary arms only, as registered.
- **Eval batch:** 8. The batch-32 comparison numbers are the **registered** ones, read from the
  original caches (`checkpoints/phase8{c,d}/eval_cache/`), which are never rewritten.
- **Reproduction gate (runs first):** baseline, scramble and oracle are re-scored at batch 32 into
  a separate cache, and the number of graph×seed cells that differ from the registered caches is
  reported. Its purpose is to separate the batch effect from any hardware/software drift since
  July–September; the scramble cache dates from the GPU outage.
- **Code:** `scripts/o19_batch8_rescore.py`. It calls the original scoring functions with their
  cache redirected to `checkpoints/o19_eval_cache/{8c,8d}_b<batch>/`. This is needed because the
  8c cache filename has no batch tag, so a plain `--batch-size 8` would silently reload the
  batch-32 results. Output: `experiments/o19_8c8d_batch8_results.{md,json}`.
- **Contrasts (each at batch 32 and batch 8):**
  1. 8c primary, CONV: channel/spatial/full − baseline (BH across the three).
  2. 8c placebo, OTHER: the same three.
  3. P4, CONV: channel − scramble. P4 placebo, OTHER: scramble − baseline.
  4. 8d, CONV: oracle − channel (primary), oracle − baseline (secondary). OTHER: oracle − baseline.
     Oracle CONV mean against the P5 threshold of 1.05, and the ceiling-recovery fraction.
- **Shared results:** the batch-8 caches for the lr-1e-3 8c baseline and channel are the
  batch-8 sensitivity cells that 8e Amendment 1 (`phase8e_lr_robustness_preregistration.md:161-166`)
  and R3 (`r3_scramble_gpu_replication_declaration.md:31`) report. Those analyses read these
  caches instead of recomputing them, so the same cell can't get two different batch-8 numbers.

## 3. Same-verdict rule (fixed now)

A row has the **same verdict** at batch 8 if Δ has the same sign **and** falls on the same side of
.05 (BH-p for the 8c primary rows, t-p for the others).

## 4. How it will be used (fixed now)

The registered batch-32 results stay **primary** whatever happens here. Batch 8 is reported beside them.

| Outcome at batch 8 | Wording / consequence |
|---|---|
| Every 8c primary row and P4 has the same verdict | Methods states that Top-1 is scored at batch 32 and the appendix eval-batch table reports batch 8. "8c/P4 verdicts are unchanged at eval batch 8." O19 → answered. |
| An 8c primary row or P4 keeps its sign but loses significance | §7 main text (not only the appendix) says that this result is significant at eval batch 32 (registered) and not at batch 8, and gives the batch-8 MDE. Finding 3 is not described as robust to the eval batch. |
| An 8c primary row or P4 flips sign | §7 reports the registered result with the batch-8 reversal next to it. Finding 3's claim is restated as eval-batch-dependent. The first author and the second author decide the wording before anything is written. |
| A placebo row changes verdict | Reported in the same table. Under 8c §8 ("OTHER moves too → confound"), it is disclosed beside the primary row it belongs to. It does not override the registered verdict. |
| 8d | Descriptive, as registered. Report the oracle CONV mean against the P5 threshold, and the recovery fraction, at both batches. If oracle − channel becomes significant at batch 8, §8 says the 8d primary is eval-batch-sensitive. |
| Reproduction gate shows differing cells | Report the count and the maximum difference. Batch-8 rows are still reported, and the drift is disclosed as a second source of difference for the arms it affects. |

No other batch sizes and no extra seeds. Batch 2 is already known to shrink every arm (O1).
**Cutoff:** if the queue has not started by 2026-10-23 23:59 ET, O19 is not run, and the paper
states batch 32 and that its sensitivity was not checked for 8c/8d.

## 5. Budget and order

- Eval-only. 12 batch-8 scoring jobs (5 arms × 2 collections, plus oracle × 2) and 6 gate jobs
  (3 arms × 2 collections).
- The original batch-32 GPU caches took about 15 min per arm × collection (cache mtimes, Jul 13
  17:30–19:30 for 8 caches). Batch 8 needs 4× as many forward passes over the same items.
  Estimate ≈ 4–10 GPU-h.
- Queue o starts only after k, l, m and n have exited and no `train*.py` or R0 process is
  running, so it never displaces 8e, R4, R3 or R0. It is the first item to drop if GPU time runs out.

## Sign-off

- [x] the first author sign-off (date): 2026-10-13 (given in session; recorded by Claude)
- [x] Frozen on commit: the commit that adds this file (with `scripts/o19_batch8_rescore.py` and the queue o script)
