# Phase 6b per-graph τ at eval batch 8 vs 32 (O21; exploratory sensitivity)

Declared in `experiments/o21_6b_batch8_declaration.md`. Batch 32 = the original `phase6b_confound_checks.json` (unchanged); batch 8 = this re-score. xla:default, lr 1e-3, 5 seeds.

## Gate: per-seed mean over the 7 graphs vs the checkpoint's stored batch-8 val τ

- wide-baseline: max |Δ| 1.69e-06
- hardware: max |Δ| 2.05e-06

## Per-graph Δτ (hardware − flat, mean over seeds)

| graph | b32 | b8 |
|---|---|---|
| bert_pretraining.4x4.fp16 | +0.204 | +0.190 |
| inception_v3_batch_128_train | +0.568 | +0.550 |
| mlperf_bert_batch_24_2x2 | -0.001 | -0.046 |
| resnet50.4x4.fp16 | +0.070 | +0.125 |
| resnet_v1_50_official_batch_128_bf16 | +0.080 | +0.168 |
| tf2_bert_pretrain_dynamic_batch_size | +0.206 | +0.186 |
| unet_3d.4x4.bf16 | -0.020 | -0.018 |

| quantity | b32 | b8 |
|---|---|---|
| inception_v3 Δ | +0.568 | +0.550 |
| mean of 7 per-graph Δ | +0.158 | +0.165 |
| mean of the other 6 | +0.090 | +0.101 |
| flat-vs-hardware paired p | 4.97e-05 | 9.95e-06 |
| BH-adjusted p (flat-vs-hier / hier-vs-hw / flat-vs-hw) | 0.0029 / 0.0299 / 0.0001 | 0.0029 / 0.0299 / 0.0000 |
| tightening, flat: median SD at n = 7 (p05–p95) | 0.0152 (0.0071–0.0549) | 0.0145 (0.0068–0.0537) |
| tightening, flat: median SD at n = 20 | 0.0135 | 0.0131 |

## Outcome (declaration §2, applied 2026-10-15)

- Gate passed (max |Δ| 2.05e-06 vs stored batch-8 val τ).
- **inception_v3's share does not change materially:** Δ +0.568 → +0.550, mean of the other 6 +0.090 → +0.101 (ratio 6.3 → 5.4; the §2 trigger is a doubling or halving). The O3 sentence quotes the batch-8 numbers (+0.550; 0.165 → 0.101 without it) with no "changed" note.
- BH results unchanged (all three contrasts significant at both batches).
- **FIG-M2:** quote batch 8 (median SD 0.0145 at n = 7, p05–p95 0.0068–0.0537; 0.0131 at n = 20). Batch 32 differs in the third decimal (0.0152 / 0.0135), so the caption gives the batch-32 value beside it. The figure currently plotted is batch 32 → regenerate from `phase6b_confound_checks_b8.json`.
- No verdict depends on this (appendix material, lr-1e-3 contrast).
