# R5 — Oracle vs baseline, both at the tuned LR (headroom after tuning), 10 seeds

> **Status: FROZEN ON COMMIT, 2026-10-16.** the first author approved it in session (2026-10-16). No R5 run
> started before this commit; any change after it is a dated amendment made before the first run.
> Origin: 8e (`phase8e_results.md`) showed the 8c baseline was LR-handicapped (tuned CONV Top-1
> 1.0904 vs 1.4246 at lr 1e-3). 8d's oracle (1.0478) was trained at lr 1e-3 only, so "how much
> conv headroom is left once the baseline is tuned" has no like-for-like test. R5 is that test.
> **It is motivated by 8e's outcome and its power analysis uses already-observed seed spreads
> (§5), so it is exploratory in origin.** What is fixed in advance is the design, the n, the
> primary contrast and the wording of every outcome (§6).

## 0. Observation status

- **Known:** 8e baseline@3e-5 and channel@3e-5 (5 seeds, CONV Top-1 1.0904 and 1.0573 at batch 32);
  8d oracle-bucket@1e-3 (5 seeds, trained on CPU, CONV 1.0478, all checkpoints from epoch 5, O23);
  8d's P5 outcome and its unsatisfiability (`phase8d_power_addendum.md`).
- **Never run:** the oracle arm at any LR other than 1e-3, or on GPU; the baseline arm at 3e-5
  with any seed outside {42, 100, 123, 456, 789}.

## 1. Question

Once the static-graph baseline is LR-tuned, does a model also given runtime-derived rank
information still rank conv layouts better? That is, is there detectable conv headroom left after
tuning that information outside the static graph recovers?

## 2. Design (fixed now)

- **Arms:**
  - `baseline`: `train_8c.py --mxu-arm baseline --lr 3e-5` (8e's LR\*, unchanged recipe).
  - `oracle-bucket`: `train_8d.py --lr 3e-5`, the 8d arm unchanged (K = 5 rank buckets,
    158 + 5 inputs, `dataset_oracle.py`). Only the LR differs from 8d.
- **LR:** 3e-5 for both, the baseline's final 8e LR\*. The oracle is **not** tuned on its own grid.
  This is deliberate. If 3e-5 under-tunes the oracle, the oracle is handicapped, so a significant
  oracle win is conservative; a null is worded to allow for it (§6).
- **Seeds (n = 10):** {42, 100, 123, 456, 789} (8e's) + {7, 21, 314, 2718, 9001} (new, fixed now).
  - baseline: the 5 existing 8e checkpoints
    (`checkpoints/phase8e/xla_random/mxu-baseline-lr3e-5-seed<s>`) are reused unchanged; 5 new runs.
    Training code is unchanged since 548f814 (2026-10-05), before the 8e runs.
  - oracle: 10 new runs.
  - **15 new runs**, all on GPU. Paths: `checkpoints/r5/xla_random/mxu-baseline-lr3e-5-seed<s>`
    and `checkpoints/r5/xla_random/oracle-bucket-lr3e-5-seed<s>`.
- **Recipe:** 8c/8e unchanged: xla:random train, 50 epochs, cosine LR → LR/100, wd 1e-4,
  grad-clip 0.1, dropout 0, `--configs-per-batch 8`, `--max-configs-per-graph 128`, capped harness
  (`launch_capped.sh`, num-workers 0). Every run's LR is checked with `scripts/check_run_lr.py`.
- **Checkpoint selection:** `best_model.pt` by overall val τ on xla/random/valid, the rule every
  arm uses. The oracle's τ is inflated by its leaked input. That is disclosed, as in 8d (O23).
- **Metric:** CONV Top-1 slowdown on 8c's frozen split (`phase8c_evaluate.py`: CONV_GRAPHS ×
  {xla/default, xla/random} valid, mean over the 6 instances per seed). **Eval batch 32 is
  primary** (as 8c/8d/8e); batch 8 is the sensitivity check. The evaluator
  (`scripts/r5_evaluate.py`) is written and committed before the first oracle checkpoint is
  scored; it reads 8e's baseline caches for the 5 existing seeds and recomputes nothing else.

## 3. Contrasts

1. **Primary:** oracle@3e-5 − baseline@3e-5, CONV, batch 32, paired by seed, n = 10. Two-sided
   paired t (decides), Wilcoxon (floor .002 at n = 10), bootstrap 95% CI, Cohen's d, realised MDE,
   per-seed table.
2. **Placebo:** the same on OTHER. Movement is expected (the bucket signal is informative for all
   graphs, 8d §8); it is reported, and an OTHER gain ≥ the CONV gain is flagged as "not
   conv-specific" (8d's rule).
3. **Descriptive only, no test decides anything:**
   - channel@3e-5 vs oracle@3e-5 on the 5 shared seeds, and recovery
     (baseline − channel) / (baseline − oracle) at the tuned LR;
   - oracle@3e-5 vs oracle@1e-3 (8d) on the 5 shared seeds (also differs in hardware, CPU vs GPU);
   - overall val τ for both arms; the epoch each oracle checkpoint was selected at;
   - baseline mean on the 5 new seeds vs the 5 8e seeds (does the 10-seed baseline differ from
     8e's 1.0904?).

## 4. What R5 does not change

8e's registered results (5 seeds) stay as they are. R5's 10-seed baseline is R5's own and is not
fed back into 8e. 8d's P5 stays "not confirmed". P4 and 8c stay lr-1e-3 results.

## 5. Power and satisfiability (checked before any run)

`power_analysis.py mde --sigma 0.0437 0.0652 --n-seeds 5 8 10` (paired t, α = .05, power .8):

| σ (paired, CONV Top-1) | Source | MDE n = 5 | MDE n = 10 |
|---|---|---|---|
| 0.0437 | 8e Q1 (channel − baseline, both at 3e-5) | 0.073 | 0.044 |
| 0.0652 | proxy: 8d oracle@1e-3 − 8e baseline@3e-5, same seeds | 0.108 | 0.065 |

- **Ceiling:** Top-1 ≥ 1.0, so the largest possible effect is 1.0904 − 1.0 = **0.090** (on 8e's 5
  seeds; the 10-seed baseline mean may differ).
- **At n = 5 the design is unsatisfiable** at the proxy σ (MDE 0.108 > 0.090). This is why n = 10.
- **At n = 10 it is satisfiable with a boundary:** 80% power requires the tuned oracle to reach
  about **≤ 1.025**. At the proxy σ, p < .05 is reached for a mean Δ of about −0.045 (oracle
  ≈ 1.045), with lower power.
- The proxy σ is driven by the baseline's two weak seeds (123: 1.159, 456: 1.156); the oracle was
  near-constant at 1e-3 (sd 0.006). If the 5 new baseline seeds are as variable, σ ≈ 0.065 holds.
- Therefore a null is read with the realised MDE, never as "no headroom".

## 6. How it will be used (fixed now)

| Primary outcome | Wording it supports | Paper consequence |
|---|---|---|
| Δ < 0, p < .05 | "With both arms LR-tuned, a model given runtime-derived rank information beats the static-graph baseline on convolution graphs by Δ (95% CI …)." | §7 ¶5–6: a like-for-like measure of conv headroom left after tuning, on 3 graphs from 2 families. §6's information gap gets a tuned model-based test. Never "the floor is representational" or "proven"; the oracle is leaky by design. |
| Δ < 0, p ≥ .05 | "No headroom detected after tuning (Δ …, MDE …): on these graphs a tuned flat model is within the MDE of a model given runtime ranks." | §7 ¶5–6 scoped as such; §6's gap claim rests on sensitivity (2.77×) and the failed static explanations, not on model error. Never "no headroom" or "the gap is gone". |
| Δ ≥ 0 | "A tuned oracle does not beat the tuned baseline on these graphs." | §6 states that the conv error observed at lr 1e-3 is not shown to reflect missing information. Registered now so it cannot be framed as a wash. |

In every row: the batch-8 result is reported beside it. If its sign differs from batch 32, that is
disclosed in the same sentence (as R4). No "trending".

## 7. Rules

1. The LR, seeds, n, arms and metric are fixed. No seed is added or replaced after the first run.
2. A run that crashes is restarted once from scratch at the same seed. If it fails again, R5 is
   reported at the n that finished, flagged, with the MDE recomputed for that n.
3. **Cutoff:** every run must finish by 2026-10-23 23:59 ET; otherwise R5 is not reported (no
   partial result).
4. If the realised CONV spread makes the result unsatisfiable after the fact (realised MDE >
   the realised ceiling), that is stated beside the verdict, as 8e §6 does.

## 8. Budget and order

15 runs × ≈ 2 h (8e/R3 GPU timing; the oracle's GPU timing is checked on its first run) ≈ 30
GPU-h, about Oct 17–18 if launched Oct 16. Order: the 5 oracle runs on 8e's seeds first (each
pairs with an existing baseline), then the 5 new seeds as baseline–oracle pairs. Queue
`scripts/run_gpu_queue_2026-10-16r5.sh`, launched detached in a systemd scope so it survives the
session, with the `PAUSE_GPU_QUEUE` switch.
