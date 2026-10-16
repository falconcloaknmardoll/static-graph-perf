# R3 — P4 at matched hardware: scramble arm re-trained on GPU (replication)

Frozen: `r3_scramble_gpu_replication_declaration.md`. lr 1e-3, 5 seeds, 8c frozen split (CONV 3 graphs × 2 xla collections, OTHER 4 × 2). **Primary = eval batch 32 (as P4)**; batch 8 = sensitivity. 8c arms read from the registered caches (b32) and O19's caches (b8). The registered P4 (scramble on CPU) stays primary; R3 is reported beside it.

## Eval batch 32 (primary)

| arm | CONV Top-1 | OTHER Top-1 |
|---|---|---|
| baseline (GPU, 8c) | 1.4246 ± 0.0721 | 1.0272 ± 0.0027 |
| channel (GPU, 8c) | 1.1203 ± 0.1077 | 1.0292 ± 0.0182 |
| scramble (CPU, 8c) | 1.3645 ± 0.0572 | 1.0351 ± 0.0039 |
| scramble (GPU, R3) | 1.3415 ± 0.1072 | 1.0344 ± 0.0186 |

| contrast (negative = first arm closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|
| R3-P4: channel − scramble-GPU, CONV | -0.2212 | 0.0033 | 0.0625 | [-0.2889, -0.1715] | -2.80 | 0.1312 |
| scramble-GPU − baseline, CONV | -0.0831 | 0.141 | 0.188 | [-0.1640, -0.0050] | -0.82 | 0.1690 |
| placebo: scramble-GPU − baseline, OTHER | +0.0072 | 0.477 | 0.625 | [-0.0086, +0.0233] | 0.35 | 0.0342 |
| hardware offset: scramble-GPU − scramble-CPU, CONV | -0.0230 | 0.611 | 0.625 | [-0.0952, +0.0492] | -0.25 | 0.1553 |
| hardware offset: scramble-GPU − scramble-CPU, OTHER | -0.0007 | 0.941 | 1 | [-0.0171, +0.0156] | -0.04 | 0.0341 |
| reference (registered P4): channel − scramble-CPU, CONV | -0.2442 | 0.00631 | 0.0625 | [-0.3154, -0.1606] | -2.35 | 0.1730 |

## Eval batch 8 (sensitivity)

| arm | CONV Top-1 | OTHER Top-1 |
|---|---|---|
| baseline (GPU, 8c) | 1.4289 ± 0.0825 | 1.0349 ± 0.0066 |
| channel (GPU, 8c) | 1.1266 ± 0.0624 | 1.0381 ± 0.0057 |
| scramble (CPU, 8c) | 1.3187 ± 0.0504 | 1.0341 ± 0.0088 |
| scramble (GPU, R3) | 1.3597 ± 0.0464 | 1.0397 ± 0.0173 |

| contrast (negative = first arm closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |
|---|---|---|---|---|---|---|
| R3-P4: channel − scramble-GPU, CONV | -0.2330 | 0.00363 | 0.0625 | [-0.2961, -0.1637] | -2.73 | 0.1417 |
| scramble-GPU − baseline, CONV | -0.0693 | 0.0649 | 0.125 | [-0.1120, -0.0180] | -1.13 | 0.1019 |
| placebo: scramble-GPU − baseline, OTHER | +0.0048 | 0.439 | 0.625 | [-0.0042, +0.0158] | 0.38 | 0.0209 |
| hardware offset: scramble-GPU − scramble-CPU, CONV | +0.0409 | 0.176 | 0.312 | [-0.0018, +0.0843] | 0.73 | 0.0926 |
| hardware offset: scramble-GPU − scramble-CPU, OTHER | +0.0056 | 0.615 | 0.812 | [-0.0123, +0.0236] | 0.24 | 0.0385 |
| reference (registered P4): channel − scramble-CPU, CONV | -0.1921 | 0.00768 | 0.0625 | [-0.2646, -0.1280] | -2.22 | 0.1438 |

## Verdict (declaration §3 table, batch 32)

- R3-P4 Δ -0.2212, p = 0.0033, MDE 0.1312 → **P4 replicates at matched hardware**.
- OTHER placebo on GPU: Δ +0.0072, p = 0.477 (b8: +0.0048, p = 0.439). The registered CPU-arm placebo was +0.0079, p = .028. It does not move on GPU, consistent with the CPU-offset explanation.

## Outcome (declaration §3 table, applied 2026-10-16)

- **Row 1 applies: "P4 replicates at matched hardware."** R3-P4 (channel, GPU − scramble, GPU, CONV, batch 32) Δ −0.2212, 95% CI [−0.2889, −0.1715], p = .0033, d −2.80, MDE 0.131. Batch 8 the same: −0.2330, p = .0036. Report it beside the registered P4 (scramble on CPU: −0.2442, p = .0063), which stays primary.
- **O17 → answered for P4.** The §7 wording changes from "not tested at matched hardware" to "replicated at matched hardware". The compute statement still says which registered arms trained on CPU (the 8c scramble control and the 8d oracle).
- **Contrast 3 (hardware offset, descriptive):** scramble GPU − CPU, CONV −0.0230 (p = .61, MDE 0.155), OTHER −0.0007 (p = .94). No detectable offset; at n = 5 this rules out only a large one.
- **Placebo:** scramble-GPU − baseline, OTHER +0.0072, p = .48 (batch 8 +0.0048, p = .44). The registered CPU-arm wobble (+0.0079, p = .028) does not recur on GPU. Together with O19 (absent at batch 8) it is attributed to the CPU arm and the eval batch, not to the scramble feature.
- Wilcoxon p = .0625 on every contrast is the n = 5 floor (all five seeds in the same direction); the t-test decides.
- Scope: like 8c and P4, an lr-1e-3 result. 8e's verdict ("not shown robust to LR tuning") still governs §7's claims; R3 says the lr-1e-3 gain is specific to the real MXU values, not that it survives tuning.
