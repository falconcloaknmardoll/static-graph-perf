# Phase 8e — results (LR robustness of the 8c channel effect)

Pre-registration: `phase8e_lr_robustness_preregistration.md` (frozen 44267c7) + Amendment 1. LR* (overall xla:random val τ, boundary rule applied): baseline **3e-5**, channel **3e-5**. CONV Top-1 at eval batch 32 (primary, Amendment 1); batch 8 beside it.

## Overall xla:random val τ by arm × LR (§5 item 1; checkpoint `val_tau`)

| arm | 3e-5 | 1e-4 | 3e-4 | 1e-3 |
|---|---|---|---|---|
| baseline | 0.4068 ± 0.0207 | 0.3967 ± 0.0095 | 0.3326 ± 0.0193 | 0.2311 ± 0.0256 |
| channel | 0.4441 ± 0.0321 | 0.4173 ± 0.0169 | 0.3969 ± 0.0114 | 0.3326 ± 0.0563 |

## Registered tests (§3, §7) — CONV Top-1, paired by seed, eval batch 32

| test | contrast | Δ | 95% CI | t-p | Holm-p | Wilcoxon-p | d | MDE |
|---|---|---|---|---|---|---|---|---|
| Q1 | channel@3e-5 − baseline@3e-5 | -0.0331 | [-0.0705, -0.0013] | 0.1664 | 0.3328 | 0.188 | -0.76 | 0.0727 |
| Q2 | channel@1e-3 − baseline@3e-5 | +0.0299 | [-0.0911, +0.1509] | 0.6938 | 0.6938 | 0.812 | +0.19 | 0.2625 |

**Q1 verdict (§7, unadjusted p): Not shown robust to LR tuning.** Q2 (same wording): The 8c advantage is attributable to LR / training dynamics.
- Satisfiability boundary (§6): tuned baseline CONV Top-1 = 1.0904 (BELOW the ≈1.16–1.19 boundary).
- Convergence (§5 item 4): runs at LR* peaking at epoch ≥ 46: baseline 1/5, channel 1/5 → no budget qualifier.
- LR* = 3e-5 / 3e-5 is the lower edge of the enlarged grid; §4 allows no further step. Disclose.

## Reported either way (§5)

| item | subset | b32 Δ | b32 p | b8 Δ | b8 p |
|---|---|---|---|---|---|
| Q1 | CONV | -0.0331 | 0.1664 | -0.0602 | 0.0446 |
| Q1 | OTHER (placebo) | +0.0033 | 0.4000 | +0.0111 | 0.0438 |
| Q2 | CONV | +0.0299 | 0.6938 | pending O19 |  |
| Q2 | OTHER (placebo) | +0.0075 | 0.5002 | pending O19 |  |

## Per-instance CONV Top-1 at LR* (§5 item 3; mean over seeds, batch 32)

| collection | graph | baseline@3e-5 | channel@3e-5 |
|---|---|---|---|
| xla/default | inception_v3_batch_128_train | 1.0061 | 1.0132 |
| xla/default | resnet50.4x4.fp16 | 1.0757 | 1.0606 |
| xla/default | resnet_v1_50_official_batch_128_bf16 | 1.0070 | 1.0460 |
| xla/random | inception_v3_batch_128_train | 1.0057 | 1.0275 |
| xla/random | resnet50.4x4.fp16 | 1.4318 | 1.1897 |
| xla/random | resnet_v1_50_official_batch_128_bf16 | 1.0160 | 1.0071 |

## CONV Spearman(pred, runtime) at LR* (§5 item 5; mean over graphs × seeds; good ranking ≪ 0)

| arm | xla/default | xla/random |
|---|---|---|
| baseline@3e-5 | -0.224 | -0.379 |
| channel@3e-5 | -0.167 | -0.512 |

## Reproduction gate: 8c lr-1e-3 cells re-scored at batch 32 vs the registered caches

- baseline xla/default: 1/35 graph×seed cells differ, max |Δ| 1.91e-08
- baseline xla/random: 0/35 graph×seed cells differ, max |Δ| 0.00e+00
- channel xla/default: 0/35 graph×seed cells differ, max |Δ| 0.00e+00
- channel xla/random: 2/35 graph×seed cells differ, max |Δ| 2.09e-06

Figure: `experiments/figures/phase8e_lr_curves.png`.
