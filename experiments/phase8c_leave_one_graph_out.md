# 8c — leave-one-graph-out (post hoc, exploratory)

Channel − baseline CONV Top-1 (batch 32), paired by seed, dropping one CONV graph (both collections) at a time. Source: `phase8c_results.json` raw. Not pre-registered.

| dropped | baseline | channel | Δ | t-p | seeds improving |
|---|---|---|---|---|---|
| none (registered) | 1.4246 | 1.1203 | -0.3043 | 0.0021 | 5/5 |
| inception_v3_batch_128_train | 1.3698 | 1.0947 | -0.2751 | 0.0005 | 5/5 |
| resnet50.4x4.fp16 | 1.3950 | 1.1353 | -0.2597 | 0.0076 | 5/5 |
| resnet_v1_50_official_batch_128_bf16 | 1.5091 | 1.1309 | -0.3783 | 0.0099 | 5/5 |

Per graph (mean over both collections and 5 seeds):

| graph | baseline | channel |
|---|---|---|
| inception_v3_batch_128_train | 1.5344 | 1.1715 |
| resnet50.4x4.fp16 | 1.4838 | 1.0902 |
| resnet_v1_50_official_batch_128_bf16 | 1.2557 | 1.0992 |

Caveat: two of the three graphs are ResNet-50 variants, so this is two architecture families. Graph-level inference stays weak; R4 (`r4_heldout_conv_declaration.md`) tests held-out families.
