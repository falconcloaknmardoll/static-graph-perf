# R5 — Oracle vs baseline, both at the tuned LR (3e-5), 10 seeds

Frozen: `r5_oracle_tuned_declaration.md` (58cb565). CONV = 8c's frozen split (3 graphs × 2 xla collections). **Primary = eval batch 32**; batch 8 = sensitivity. n = 10. Baseline on 8e's seeds, channel@3e-5 and oracle@1e-3 are read from their registered caches.

## Eval batch 32 (primary)

| arm (lr 3e-5) | CONV Top-1 | OTHER Top-1 |
|---|---|---|
| baseline | 1.0852 ± 0.0575 | 1.0195 ± 0.0072 |
| oracle-bucket | 1.0420 ± 0.0077 | 1.0398 ± 0.0020 |

| contrast (negative = first arm closer to optimal) | n | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|---|
| primary: oracle − baseline, CONV | 10 | -0.0432 | 0.0564 | 0.105 | [-0.0812, -0.0086] | -0.69 | 0.0621 |
| placebo: oracle − baseline, OTHER | 10 | +0.0204 | 2.26e-05 | 0.00195 | [+0.0158, +0.0253] | 2.52 | 0.0080 |

Satisfiability (§7.4): realised ceiling 0.0852 vs realised MDE 0.0621 → satisfiable.

Per seed (CONV Top-1):

| seed | baseline | oracle | Δ |
|---|---|---|---|
| 42 | 1.0659 | 1.0420 | -0.0240 |
| 100 | 1.0369 | 1.0535 | +0.0166 |
| 123 | 1.1588 | 1.0259 | -0.1329 |
| 456 | 1.1563 | 1.0403 | -0.1160 |
| 789 | 1.0341 | 1.0419 | +0.0079 |
| 7 (new) | 1.0365 | 1.0535 | +0.0171 |
| 21 (new) | 1.0969 | 1.0420 | -0.0549 |
| 314 (new) | 1.1758 | 1.0403 | -0.1355 |
| 2718 (new) | 1.0503 | 1.0403 | -0.0100 |
| 9001 (new) | 1.0408 | 1.0403 | -0.0005 |

Descriptive (§3.3, 8e's seeds, n = 5; no verdict):

| contrast (negative = first arm closer to optimal) | n | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|---|
| channel@3e-5 − oracle@3e-5, CONV | 5 | +0.0166 | 0.341 | 0.438 | [-0.0063, +0.0461] | 0.48 | 0.0572 |
| oracle@3e-5 − oracle@1e-3 (8d; also GPU vs CPU), CONV | 5 | -0.0071 | 0.238 | 0.812 | [-0.0175, +0.0002] | -0.62 | 0.0191 |

- Tuned recovery (baseline − channel) / (baseline − oracle): 0.665
- Baseline CONV, 8e seeds 1.0904 vs new seeds 1.0800

## Eval batch 8 (sensitivity)

| arm (lr 3e-5) | CONV Top-1 | OTHER Top-1 |
|---|---|---|
| baseline | 1.1160 ± 0.0685 | 1.0134 ± 0.0043 |
| oracle-bucket | 1.0233 ± 0.0054 | 1.0325 ± 0.0048 |

| contrast (negative = first arm closer to optimal) | n | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|---|
| primary: oracle − baseline, CONV | 10 | -0.0926 | 0.00261 | 0.00977 | [-0.1333, -0.0497] | -1.30 | 0.0708 |
| placebo: oracle − baseline, OTHER | 10 | +0.0190 | 3.52e-06 | 0.00195 | [+0.0154, +0.0225] | 3.17 | 0.0060 |

Satisfiability (§7.4): realised ceiling 0.1160 vs realised MDE 0.0708 → satisfiable.

Per seed (CONV Top-1):

| seed | baseline | oracle | Δ |
|---|---|---|---|
| 42 | 1.1038 | 1.0226 | -0.0812 |
| 100 | 1.1558 | 1.0202 | -0.1356 |
| 123 | 1.1655 | 1.0202 | -0.1453 |
| 456 | 1.1931 | 1.0226 | -0.1705 |
| 789 | 1.1633 | 1.0202 | -0.1432 |
| 7 (new) | 1.0339 | 1.0373 | +0.0033 |
| 21 (new) | 1.1042 | 1.0272 | -0.0769 |
| 314 (new) | 1.0329 | 1.0202 | -0.0127 |
| 2718 (new) | 1.0149 | 1.0226 | +0.0077 |
| 9001 (new) | 1.1921 | 1.0202 | -0.1719 |

Descriptive (§3.3, 8e's seeds, n = 5; no verdict):

| contrast (negative = first arm closer to optimal) | n | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|---|
| channel@3e-5 − oracle@3e-5, CONV | 5 | +0.0750 | 0.0531 | 0.0625 | [+0.0266, +0.1233] | 1.22 | 0.1026 |
| oracle@3e-5 − oracle@1e-3 (8d; also GPU vs CPU), CONV | 5 | -0.0068 | 0.148 | 0.188 | [-0.0137, -0.0005] | -0.80 | 0.0142 |

- Tuned recovery (baseline − channel) / (baseline − oracle): 0.445
- Baseline CONV, 8e seeds 1.1563 vs new seeds 1.0756

## Checkpoints (selected by overall val τ; the oracle's τ is inflated by its leaked input, O23)

| seed | baseline epoch | baseline val τ | oracle epoch | oracle val τ |
|---|---|---|---|---|
| 42 | 40 | 0.4302 | 35 | 0.7877 |
| 100 | 50 | 0.3910 | 40 | 0.7852 |
| 123 | 35 | 0.4185 | 35 | 0.7854 |
| 456 | 45 | 0.4142 | 15 | 0.7841 |
| 789 | 45 | 0.3799 | 30 | 0.7822 |
| 7 | 45 | 0.4116 | 20 | 0.7884 |
| 21 | 45 | 0.4262 | 30 | 0.7913 |
| 314 | 35 | 0.3942 | 20 | 0.7867 |
| 2718 | 35 | 0.4077 | 25 | 0.7897 |
| 9001 | 45 | 0.3976 | 20 | 0.7888 |

## Verdict (declaration §6, batch 32)

- Δ -0.0432, 95% CI [-0.0812, -0.0086], p = 0.0564, MDE 0.0621, n = 10 → **no headroom detected after tuning**.
- Batch 8: Δ -0.0926, p = 0.00261.

## Outcome (declaration §6 table, applied 2026-10-19)

- **Row 2 applies: "No headroom detected after tuning (Δ −0.0432, MDE 0.0621): on these graphs a tuned flat model is within the MDE of a model given runtime ranks."** Oracle − baseline, CONV, both at 3e-5, batch 32, n = 10: 95% CI [−0.0812, −0.0086], t-p = .056 (decides), Wilcoxon .105, d −0.69. Never "no headroom", "the gap is gone" or "trending".
- **The bootstrap CI excludes zero while the deciding t-test does not reach .05.** The CI is a percentile bootstrap (10,000 resamples of the 10 per-seed deltas, `phase8c_evaluate.paired_stats`), which runs narrow at n = 10. The t-interval matching the deciding test is [−0.0879, +0.0015] and includes zero (batch 8: [−0.1435, −0.0417]; computed 2026-10-19 from the per-seed deltas above). The registered decision is the t-test; in the text give Δ, p and MDE, and label any CI by method.
- **Batch 8 (reported beside it, same sign):** Δ −0.0926, p = .0026, MDE 0.071. The sign agrees with batch 32, so no flip to disclose (§6), but the batch-8 result is significant: the oracle contrast is eval-batch-sensitive, as 8d's was (O19).
- **Satisfiability (§7.4):** realised ceiling 0.0852 > realised MDE 0.0621 (batch 8: 0.1160 > 0.0708), so the test could have detected an effect of this size. State it beside the verdict.
- **Placebo (§3.2):** OTHER oracle − baseline +0.0204, p < .001 (batch 8 +0.0190, p < .001): the tuned oracle is *further* from optimal than the tuned baseline on non-conv graphs. The "not conv-specific" flag (OTHER gain ≥ CONV gain) does not fire, since this is a loss, not a gain. Disclose: the bucket input helps CONV only directionally and costs on OTHER.
- **Concentration:** three of ten seeds carry most of the batch-32 effect (123 −0.133, 314 −0.136, 456 −0.116; together −0.384 of the −0.432 mean sum); six sit within ±0.025. The baseline's new seeds match 8e's (1.0800 vs 1.0904).
- **Descriptive only (§3.3):** channel@3e-5 − oracle@3e-5 +0.0166 (p = .34, n = 5); tuned recovery 0.665 (batch 8: 0.445); oracle@3e-5 ≈ oracle@1e-3 (−0.0071, p = .24). Oracle checkpoints now peak at epochs 15–40 (8d's were all epoch 5, O23).
- **Consequence for the paper (§6 row 2):** §7 ¶5–6 are scoped as above. §6's information-gap claim rests on the model-free sensitivity (2.77×) and the failed static explanations, not on model error. 8e, 8d (P5 not confirmed) and P4 are unchanged (§4).
- Run note: the first evaluation died in a machine crash on 2026-10-19 00:46 before writing any cache; it was re-run from scratch (same committed evaluator, 71ffa73) under a 24 G memory cap and finished 03:40.
