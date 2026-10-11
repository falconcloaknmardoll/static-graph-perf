# Phase 6d §10.5 — Amended results: no hardware advantage on nlp:default at a matched LR

> Evaluated 2026-10-11 strictly against `phase6d_nlp_default_preregistration.md` §10.5 and
> Amendment 2 (§11). Script: `scripts/phase6d_amended_evaluate.py` → `phase6d_amended_results.json`.
> Every number is the `val_tau` stored in each run's `best_model.pt`, eval batch 8.
> **Label: AMENDED / exploratory.** The LR was chosen after diagnostic runs. This never replaces
> §10.4, the registered result: the shared lr-1e-3 recipe does not train the hardware arm on
> nlp:default (a training failure).

## Completion and validity checks

- **All 13 runs** carry `.done` markers before the 2026-10-23 cutoff (last: Oct 11 00:16), so
  §11.2 is met and the contrast may be computed.
- **LR** was confirmed from every run's epoch-1 line (`logs/gpu_queue_2026-10-05j.log`: 13 × OK), and
  checkpoint args agree (1e-4 / 1e-3).
- **Fast evaluator** was used in all 10 amended runs and flat@1e-3 seeds 123/456/789. Flat@1e-3 seeds
  42/100 were selected with `train.evaluate` (the disclosed mixing, §11.1).
- **Collapse rule:** no collapse. The final `val_n_graphs` is 20 for all 15 runs.

## Arms (n = 5 seeds: 42, 100, 123, 456, 789)

| Arm | mean τ | sd | per seed | best epoch |
|---|---|---|---|---|
| hardware @ 1e-4 | 0.4087 | 0.0119 | .4170 .4169 .3884 .4127 .4085 | 45 45 40 50 50 |
| flat @ 1e-4 | 0.4138 | 0.0064 | .4124 .4081 .4208 .4201 .4074 | 40 50 50 35 50 |
| flat @ 1e-3 | 0.3192 | 0.0503 | .2543 .3451 .3291 .3824 .2853 | 50 ×5 |

## Verdict (§7 read as A-P1, row "Δτ < 0 (any p)"): **A-P1 FAILS and A-P2 FAILS (amended)**

| Contrast | mean Δτ | sd | t | p | Wilcoxon p | bootstrap 95% CI | d_z | realized MDE |
|---|---|---|---|---|---|---|---|---|
| **A-P1** hardware@1e-4 − flat@1e-4 | **−0.0051** | 0.0164 | −0.69 | **0.527** | 1.0 | [−0.019, +0.006] | −0.31 | 0.027 |
| Confound check: hardware@1e-4 − flat@1e-3 | +0.0894 | 0.0530 | 3.78 | 0.0195 | 0.0625 | [+0.050, +0.134] | 1.69 | 0.088 |

Per-seed Δ, A-P1: +.0046 +.0087 −.0324 −.0074 +.0011. Confound check: +.1627 +.0718 +.0592 +.0303 +.1232.

- **A-P1:** the point estimate is negative, so §7's Δτ < 0 row applies whatever the p-value.
  Registered wording: A-P1 fails and A-P2 fails, because the sign patterns with nlp:random, not
  xla:default. **Both are reported, without softening.**
- **What the data can and cannot say:** with both arms at the same LR, no difference is detected,
  and the realized MDE is small (0.027). This is a well-powered null, unlike 6d's original
  ≈ 0.10 prior. The negative sign rests on a −0.005 point estimate (p = .53), so the registered
  "model family, not distribution" reading should be stated as the registered row, **not**
  strengthened into a demonstrated family effect.
- **Confound check:** it passes (p = .0195), but it only matters when A-P1 passes. It is reported
  because it shows the same pattern as 6e: against a flat arm at lr 1e-3, the hardware arm would
  have appeared to win by +0.089 (p = .02). At a matched LR the advantage is gone.

## Reported alongside (descriptive, post hoc, not tests)

1. **Flat at lr 1e-3 is handicapped on nlp:default too.** flat@1e-4 0.4138 vs flat@1e-3 0.3192
   (+0.095), and all five flat@1e-3 runs peak at epoch 50. Post hoc, n = 1 collection. It matches
   6e's xla:default pattern (flat 0.0327 → 0.2255). On the two *random* collections, flat at 1e-3
   trains normally. Not tested.
2. **Convergence:** best epoch ≥ 46 in 2/5 hardware@1e-4 and 3/5 flat@1e-4 runs. By the 6f rule
   (≥ 3/5 of an arm), the verdict carries "within a 50-epoch budget".
3. **LR selection:** 1e-4 was not chosen by a grid rule (unlike 6e/6f). It was chosen to rescue
   the hardware arm (Amendment 1). Both arms share it, so A-P1 is LR-matched but not tuned for
   either arm.

## What this does to the paper (option A)

- **No labelled exception is needed.** On nlp:default at a matched LR, the structured (hardware)
  model does not beat the flat model (Δτ −0.0051, p = .53, MDE 0.027, amended). Option A's
  "no structured model beats a matched flat model" now holds on **all four collections**: three
  registered, plus nlp:default amended/exploratory.
- **Scope wording:** symmetric LR tuning on xla:default only. LR-matched (1e-4, not tuned) on
  nlp:default. One shared lr 1e-3 on xla:random and nlp:random.
- **§10.4 still leads the nlp:default paragraph:** the registered result is a training failure under
  the shared recipe. §10.5 follows, labelled amended.
- O10 ("nlp:default cell missing") is now answered in amended form.
