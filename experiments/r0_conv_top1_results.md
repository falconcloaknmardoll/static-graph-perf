# R0 — CONV-subset Top-1 of structured arms on xla:random (exploratory, partially observed)

Frozen: `r0_conv_top1_structure_declaration.md` + Amendment 1. Checkpoints: Phase 2c/3 xla:random, lr 1e-3, 5 seeds. Split: 8c frozen (CONV 3 × 2 collections, OTHER 4 × 2). **Primary = eval batch 32 (8c's); batch 8 = sensitivity.**

## Eval batch 32 (primary)

| arm | CONV Top-1 | OTHER Top-1 |
|---|---|---|
| flat | 1.2757 ± 0.1614 | 1.0261 ± 0.0071 |
| hierarchical | 1.2972 ± 0.0643 | 1.0159 ± 0.0054 |
| hardware | 1.2783 ± 0.0742 | 1.0402 ± 0.0263 |

| contrast (negative = closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|
| hierarchical − flat, CONV | +0.0216 | 0.77 | 0.812 | [-0.0923, +0.1434] | 0.14 | 0.2566 |
| hierarchical − flat, OTHER | -0.0102 | 0.109 | 0.188 | [-0.0190, -0.0015] | -0.92 | 0.0185 |
| hardware − flat, CONV | +0.0026 | 0.974 | 1 | [-0.1391, +0.1293] | 0.02 | 0.2825 |
| hardware − flat, OTHER | +0.0141 | 0.339 | 0.312 | [-0.0058, +0.0373] | 0.49 | 0.0482 |

## Eval batch 8 (sensitivity)

| arm | CONV Top-1 | OTHER Top-1 |
|---|---|---|
| flat | 1.2454 ± 0.1383 | 1.0316 ± 0.0081 |
| hierarchical | 1.2504 ± 0.0560 | 1.0271 ± 0.0146 |
| hardware | 1.2514 ± 0.0542 | 1.0406 ± 0.0051 |

| contrast (negative = closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|
| hierarchical − flat, CONV | +0.0050 | 0.935 | 0.812 | [-0.1046, +0.0929] | 0.04 | 0.2125 |
| hierarchical − flat, OTHER | -0.0044 | 0.451 | 1 | [-0.0147, +0.0025] | -0.37 | 0.0197 |
| hardware − flat, CONV | +0.0060 | 0.932 | 1 | [-0.1139, +0.1159] | 0.04 | 0.2460 |
| hardware − flat, OTHER | +0.0090 | 0.0537 | 0.125 | [+0.0032, +0.0151] | 1.21 | 0.0124 |

Reference (not a test; different harness, see declaration §3): 8c channel − baseline CONV Δ −0.3043, BH-p .0032 (`phase8c_results.md:12`, batch 32).

## Outcome (declaration §4 table, applied 2026-10-15)

- **Row 1 applies.** Both structured − flat CONV contrasts are n.s. with |Δ| far below 8c's 0.3043 (batch 32: hierarchical +0.0216, p = .77, MDE 0.257; contention-edge +0.0026, p = .97, MDE 0.283; batch 8 the same: +0.0050 / +0.0060, MDE 0.213 / 0.246).
- Supported wording: "On xla:random at one shared recipe (lr 1e-3), adding structure does not narrow the CONV gap; adding MXU information does." Quote the MDE (≈ 0.26–0.28, close to 8c's effect size: R0 rules out an effect of 8c's size only marginally, so say "no detectable narrowing", not "no effect").
- Scope (from 8e, 2026-10-14): the MXU half is at lr 1e-3 and was not shown robust to LR tuning. R0 makes the two halves like-for-like on metric, collection and recipe; it does not change 8e's verdict.
- Labels: exploratory, partially observed (declaration §2). OTHER placebo: no row significant (contention-edge OTHER b8 +0.009, p = .054).
- Also answers O21's Phase 5 item: batch-8 per-graph Top-1 for flat and contention-edge on the same checkpoints is in `r0_conv_top1_results.json`.
