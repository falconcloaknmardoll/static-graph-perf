# Phase 6f — Results: no difference detected once both arms are LR-tuned

> Evaluated 2026-10-09 strictly against `phase6f_hardware_lr_preregistration.md` (frozen
> 2026-10-05, `9b11a96`). Script: `scripts/phase6f_evaluate.py` → `phase6f_results.json`. Every
> number below is the `val_tau` stored in each run's `best_model.pt` (the §4 selection rule),
> eval batch 8. All 20 new runs were `.done` on 2026-10-08, before the Oct 15 cutoff (§8.6).

## Verdict (§7 table, row 3): **no difference detected under symmetric tuning (n = 5), within a 50-epoch budget**

Structure is **not shown to help** on `xla:default` once both arms are LR-tuned. No
"trending" wording and no equivalence claim: the realized MDE is **0.037**, so a true difference
smaller than that cannot be detected at n = 5. This removes 6e's "the structured arm was never
tuned" caveat from Finding 1. 6e's P1 failure is still reported as registered.

## Hardware LR curve next to flat (5 seeds each; boundary step at 3e-5 fired because LR\*\_hw = 1e-4 was a grid edge)

| LR | hardware mean τ | sd | hardware per seed (42, 100, 123, 456, 789) | best epochs | flat mean τ (6e) |
|---|---|---|---|---|---|
| 3e-5 | 0.2024 | 0.0069 | .2109 .2054 .2035 .2001 .1922 | 40 45 40 45 40 | 0.2067 |
| **1e-4 = LR\*\_hw** | **0.2271** | 0.0101 | .2409 .2126 .2255 .2268 .2298 | 45 50 50 50 50 | **0.2255 = LR\*\_flat** |
| 3e-4 | 0.1898 | 0.0629 | .2138 .2224 .0778 .2115 .2237 | 45 50 50 50 45 | 0.1841 |
| 1e-3 (Phase 6) | 0.1977 | 0.0133 | .1930 .2118 .1959 .1788 .2090 | 45 20 50 45 50 | 0.0327 |
| 3e-3 | 0.1493 | 0.0796 | .1906 .1792 .1680 .2002 .0086 | 50 50 50 50 50 | 0.0261 |

LR\*\_hw = 1e-4 after the boundary step (no longer an edge of the enlarged grid). Both arms peak
at the same LR. No collapses: every new run's final `evaluate()` scored all 7 graphs. Two
low single runs (3e-4 seed 123: 0.078; 3e-3 seed 789: 0.009) are not collapses and enter the
means unchanged (§4, §8.1). They do not affect LR\*\_hw.

## Q1 (primary): hardware@1e-4 − flat@1e-4

| mean Δτ | sd | t | p | Wilcoxon p | bootstrap 95% CI | d_z | realized MDE |
|---|---|---|---|---|---|---|---|
| **+0.0016** | 0.0224 | 0.16 | **0.879** | 0.81 | [−0.017, +0.017] | 0.07 | **0.037** |

Per-seed Δ: −.0320 −.0070 +.0113 +.0084 +.0275. Flat@1e-4 per seed: .2728 .2195 .2142 .2184 .2023.

## Q2 (descriptive): hardware@1e-4 − hardware@1e-3 = **+0.029**

Tuning moved the hardware arm by +0.029, against flat's +0.193. That matches the §6 statement
that a +0.08 jump for hardware was not expected.

## Convergence check (§4) — triggers

Best epoch ≥ 46: **hardware@1e-4 4/5**, **flat@1e-4 5/5**. Both meet the ≥ 3/5 threshold, so the
verdict carries **"within a 50-epoch budget"**, with these counts reported next to it.

## Registered caveats that must travel with the result

1. **Selection bias in both arms (§4):** both LR\*s were chosen on the scoring validation set.
   The bias is disclosed, not corrected. Here both arms land on the same LR.
2. **Non-significant ≠ equivalent:** quote MDE 0.037 next to the result (§6).
3. **6b decomposition stays scoped to a shared lr of 1e-3.** `hierarchical` was not tuned (§4).
4. **The epoch budget was not tuned.** Both arms still peak at or near epoch 50 at LR\* (see the
   convergence check above).
5. **6e is reported as registered**: hardware@1e-3 − flat@1e-4 = −0.028, p = .14, P1 FAILS.
