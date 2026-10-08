# Phase 6e — Results: the xla:default win does NOT survive tuning the flat baseline's LR

> Evaluated 2026-10-08 strictly against `phase6e_flat_lr_preregistration.md` (frozen 2026-09-27).
> Script: `scripts/phase6e_evaluate.py` → `phase6e_results.json`. Every number below is the
> `val_tau` stored in each run's `best_model.pt` (the §4 selection rule), eval batch 8.

## Verdict (§7 table, row 4): **Δτ ≤ 0 → P1 FAILS**

The `xla:default` advantage is attributable to the flat arm's LR handicap at lr 1e-3. Per the
registered rule, Finding 1 is **restated as a trainability result** and is not reported as
information content. P2 fails the same way.

## Flat LR curve (5 seeds each; boundary step at 3e-5 fired because LR* = 1e-4 was a grid edge)

| Flat LR | mean τ | sd | per seed (42, 100, 123, 456, 789) |
|---|---|---|---|
| 3e-5 | 0.2067 | 0.0106 | .2154 .2040 .1894 .2124 .2123 |
| **1e-4 = LR\*** | **0.2255** | 0.0274 | .2728 .2195 .2142 .2184 .2023 |
| 3e-4 | 0.1841 | 0.0497 | .1773 .2710 .1660 .1491 .1569 |
| 1e-3 (registered) | 0.0327 | 0.0135 | .0269 .0310 .0213 .0285 .0561 |
| 3e-3 | 0.0261 | 0.0027 | .0298 .0246 .0272 .0260 .0227 |

LR\* = 1e-4 after the boundary step (no longer an edge of the enlarged grid). No collapses
(all 20 new runs evaluated all 7 graphs).

## Contrasts (structured arms at lr 1e-3, untuned, as registered)

| | mean Δτ | sd | t | p | p (Holm) | Wilcoxon p | bootstrap 95% CI | d_z | realized MDE |
|---|---|---|---|---|---|---|---|---|---|
| **P1** hardware − flat@1e-4 | **−0.0278** | 0.0337 | −1.84 | 0.139 | 0.139 | 0.125 | [−0.056, −0.004] | −0.82 | 0.056 |
| **P2** hierarchical − flat@1e-4 | **−0.0770** | 0.0283 | −6.09 | **0.0037** | **0.0074** | 0.0625 | [−0.097, −0.052] | −2.72 | 0.047 |

Per-seed Δ, P1: −.0799 −.0078 −.0183 −.0396 +.0068. P2: −.0986 −.0992 −.0328 −.0891 −.0654.

- **P1:** the sign is negative, so §7's row 4 applies whatever the p-value. The t-test is not
  significant (p = .14); the percentile bootstrap CI excludes zero on the negative side. Report
  both; do not call hardware "significantly worse" from the bootstrap alone.
- **P2:** the untuned hierarchical arm is **significantly below** the tuned flat arm (survives Holm).
- **P3 (descriptive):** flat@LR\* = **0.2255**, above TPUGraphs' published best-baseline
  validation τ of 0.12 (protocols differ, §1). The tuned baseline is no longer weak, which
  answers O5's strawman concern in the opposite direction from Finding 1.

## Registered caveats that must travel with the result

1. **Bias registered in the flat arm's favour (§4):** LR\* chosen on the scoring validation set,
   and only the flat arm was tuned. So this failure does **not** show structure is useless: the
   structured arms were never tuned. **Phase 6f** (`phase6f_hardware_lr_preregistration.md`)
   tunes the hardware arm on the same grid and rule; its result decides the fair comparison.
2. **Convergence:** all five flat@1e-4 runs reached their best τ at epoch 50, the last epoch
   (.2728@50, .2195@50, .2142@50, .2184@50, .2023@50). Disclose "within a 50-epoch budget".
3. **Edge component unaffected:** hardware − hierarchical (+0.0493, p = .030) involves no flat arm.
4. The Phase 6b decomposition (~70% hierarchy) holds only **at a shared lr of 1e-3**.
