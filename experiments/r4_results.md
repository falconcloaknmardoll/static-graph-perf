# R4 — Channel effect on held-out convolution families (exploratory)

Frozen: `r4_heldout_conv_declaration.md` + Amendment 1. Both arms at their 8e LR* = 3e-5, trained on 58 xla:random train graphs (11 conv graphs from 6 families held out), 5 seeds. Evaluation: 19 held-out instances (11 xla:random + 8 xla:default train-split files). **Primary = eval batch 32**; batch 8 = sensitivity. Read beside 8e Q1 (the tuned contrast), not 8c's lr-1e-3 contrast.

## Eval batch 32 (primary)

| arm | held-out CONV Top-1 (19 instances) | OTHER Top-1 (8c placebo) |
|---|---|---|
| baseline | 1.2963 ± 0.1284 | 1.0299 ± 0.0121 |
| channel | 1.1835 ± 0.1661 | 1.0242 ± 0.0054 |

| contrast (negative = first arm closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|
| primary: channel − baseline, held-out CONV | -0.1128 | 0.346 | 0.438 | [-0.2955, +0.0699] | -0.48 | 0.3932 |
| placebo: channel − baseline, OTHER | -0.0057 | 0.441 | 0.625 | [-0.0175, +0.0053] | -0.38 | 0.0248 |

Per family (mean over its instances, then over seeds) and leave-one-family-out:

| family | instances | baseline | channel | Δ | drop this family: Δ, t-p |
|---|---|---|---|---|---|
| alexnet | 2 | 1.8908 | 1.4457 | -0.4451 | -0.0737, 0.394 |
| maskrcnn | 3 | 1.0038 | 1.0044 | +0.0006 | -0.1341, 0.346 |
| ssd | 4 | 1.0081 | 1.0029 | -0.0052 | -0.1415, 0.348 |
| mnasnet | 2 | 1.0802 | 1.1887 | +0.1085 | -0.1389, 0.29 |
| retinanet | 6 | 1.5060 | 1.2862 | -0.2198 | -0.0635, 0.677 |
| xception | 2 | 1.3039 | 1.2376 | -0.0663 | -0.1183, 0.351 |

Per instance (mean over seeds):

| instance | baseline | channel | Δ |
|---|---|---|---|
| xla/default · alexnet_train_batch_32 | 1.0024 | 1.0008 | -0.0016 |
| xla/default · mlperf_ssd_1_shard_batch_8_fast_epoch | 1.0263 | 1.0056 | -0.0207 |
| xla/default · mlperf_ssd_2_shard_batch_8_fast_epoch | 1.0009 | 1.0006 | -0.0003 |
| xla/default · mnasnet_a1_batch_128 | 1.0657 | 1.0637 | -0.0020 |
| xla/default · retinanet.2x2.fp32 | 1.0029 | 1.0034 | +0.0005 |
| xla/default · retinanet.4x4.bf16.performance | 1.0435 | 1.0002 | -0.0433 |
| xla/default · retinanet.4x4.fp32 | 1.0090 | 1.0098 | +0.0008 |
| xla/default · xception_imagenet | 1.0842 | 1.0040 | -0.0802 |
| xla/random · alexnet_train_batch_32 | 2.7792 | 1.8907 | -0.8885 |
| xla/random · mlperf_maskrcnn_1_shard_batch_4 | 1.0002 | 1.0002 | +0.0000 |
| xla/random · mlperf_maskrcnn_batch_2 | 1.0010 | 1.0028 | +0.0018 |
| xla/random · mlperf_maskrcnn_batch_4 | 1.0101 | 1.0101 | -0.0000 |
| xla/random · mlperf_ssd_1_shard_batch_8_fast_epoch | 1.0051 | 1.0051 | +0.0000 |
| xla/random · mlperf_ssd_2_shard_batch_8_fast_epoch | 1.0001 | 1.0001 | +0.0000 |
| xla/random · mnasnet_a1_batch_128 | 1.0948 | 1.3137 | +0.2190 |
| xla/random · retinanet.2x2.fp32 | 2.2480 | 2.5244 | +0.2764 |
| xla/random · retinanet.4x4.bf16.performance | 2.7324 | 1.1795 | -1.5529 |
| xla/random · retinanet.4x4.fp32 | 1.0002 | 1.0002 | -0.0001 |
| xla/random · xception_imagenet | 1.5236 | 1.4712 | -0.0524 |

## Eval batch 8 (sensitivity)

| arm | held-out CONV Top-1 (19 instances) | OTHER Top-1 (8c placebo) |
|---|---|---|
| baseline | 1.2873 ± 0.1329 | 1.0166 ± 0.0047 |
| channel | 1.4185 ± 0.2280 | 1.0186 ± 0.0094 |

| contrast (negative = first arm closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|
| primary: channel − baseline, held-out CONV | +0.1311 | 0.305 | 0.312 | [-0.0609, +0.3151] | 0.53 | 0.4151 |
| placebo: channel − baseline, OTHER | +0.0020 | 0.703 | 0.625 | [-0.0078, +0.0081] | 0.18 | 0.0182 |

Per family (mean over its instances, then over seeds) and leave-one-family-out:

| family | instances | baseline | channel | Δ | drop this family: Δ, t-p |
|---|---|---|---|---|---|
| alexnet | 2 | 1.0434 | 2.0760 | +1.0325 | +0.0251, 0.816 |
| maskrcnn | 3 | 1.0041 | 1.0041 | +0.0000 | +0.1557, 0.305 |
| ssd | 4 | 1.1701 | 1.4592 | +0.2891 | +0.0890, 0.561 |
| mnasnet | 2 | 1.0727 | 1.1915 | +0.1188 | +0.1326, 0.326 |
| retinanet | 6 | 1.7419 | 1.5746 | -0.1673 | +0.2689, 0.0417 |
| xception | 2 | 1.0413 | 1.0596 | +0.0184 | +0.1444, 0.298 |

## Verdict (declaration §4 table, batch 32)

- Δ -0.1128, p = 0.346, MDE 0.3932 → **direction consistent on held-out families, not significant at n = 5**.
