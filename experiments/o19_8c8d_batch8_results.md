# O19 — 8c/8d CONV Top-1 at eval batch 8 vs 32 (exploratory sensitivity)

Declared in `experiments/o19_8c8d_batch8_declaration.md`. Batch 32 = registered numbers (original caches); batch 8 = this re-score. Δ < 0 means the first arm has lower slowdown.

## Reproduction gate (batch 32 re-run vs registered cache)

- baseline xla/default: 1/35 graph×seed cells differ, max |Δ| 1.91e-08
- baseline xla/random: 0/35 graph×seed cells differ, max |Δ| 0.00e+00
- scramble xla/default: 1/35 graph×seed cells differ, max |Δ| 1.24e-05
- scramble xla/random: 4/35 graph×seed cells differ, max |Δ| 4.12e-06
- oracle xla/default: 0/35 graph×seed cells differ, max |Δ| 0.00e+00
- oracle xla/random: 0/35 graph×seed cells differ, max |Δ| 0.00e+00

## Per-arm means (mean ± SD over seeds)

| arm | CONV b32 | CONV b8 | OTHER b32 | OTHER b8 |
|---|---|---|---|---|
| baseline | 1.4246 ± 0.0721 | 1.4289 ± 0.0825 | 1.0272 ± 0.0027 | 1.0349 ± 0.0066 |
| channel | 1.1203 ± 0.1077 | 1.1266 ± 0.0624 | 1.0292 ± 0.0182 | 1.0381 ± 0.0057 |
| spatial | 1.1758 ± 0.0389 | 1.1061 ± 0.0779 | 1.0237 ± 0.0080 | 1.0327 ± 0.0110 |
| full | 1.1933 ± 0.1479 | 1.1284 ± 0.1357 | 1.0125 ± 0.0080 | 1.0200 ± 0.0116 |
| scramble | 1.3645 ± 0.0572 | 1.3187 ± 0.0504 | 1.0351 ± 0.0039 | 1.0341 ± 0.0088 |
| oracle | 1.0478 ± 0.0064 | 1.0280 ± 0.0085 | 1.0294 ± 0.0146 | 1.0323 ± 0.0027 |

## Contrasts (paired by seed; two-sided t, Wilcoxon, bootstrap CI, BH over the 3 8c primary arms)

| contrast | subset | b32 Δ | b32 p (BH) | b8 Δ | b8 p (BH) | b8 CI95 | b8 MDE | same verdict? |
|---|---|---|---|---|---|---|---|---|
| 8c primary: channel − baseline | CONV | -0.3043 | 0.00322 | -0.3023 | 0.00464 | [-0.3738, -0.1959] | 0.1967 | yes |
| 8c primary: spatial − baseline | CONV | -0.2488 | 0.00137 | -0.3228 | 0.00286 | [-0.3897, -0.2628] | 0.1376 | yes |
| 8c primary: full − baseline | CONV | -0.2313 | 0.0073 | -0.3006 | 0.00464 | [-0.3939, -0.2072] | 0.1955 | yes |
| 8c placebo: channel − baseline | OTHER | +0.0020 | 0.817 | +0.0032 | 0.47 | [-0.0037, +0.0101] | 0.0151 | yes |
| 8c placebo: spatial − baseline | OTHER | -0.0035 | 0.458 | -0.0022 | 0.362 | [-0.0060, +0.0013] | 0.0079 | yes |
| 8c placebo: full − baseline | OTHER | -0.0147 | 0.0199 | -0.0149 | 0.047 | [-0.0243, -0.0059] | 0.0195 | yes |
| P4: channel − scramble | CONV | -0.2442 | 0.00631 | -0.1921 | 0.00768 | [-0.2646, -0.1280] | 0.1438 | yes |
| P4 placebo: scramble − baseline | OTHER | +0.0079 | 0.0283 | -0.0008 | 0.89 | [-0.0102, +0.0086] | 0.0204 | **NO** |
| 8d primary: oracle − channel | CONV | -0.0725 | 0.19 | -0.0986 | 0.0199 | [-0.1434, -0.0538] | 0.0976 | **NO** |
| 8d secondary: oracle − baseline | CONV | -0.3768 | 0.000276 | -0.4009 | 0.000325 | [-0.4575, -0.3301] | 0.1295 | yes |
| 8d placebo: oracle − baseline | OTHER | +0.0022 | 0.705 | -0.0026 | 0.439 | [-0.0077, +0.0025] | 0.0111 | **NO** |

## 8d descriptives

- Oracle CONV Top-1 (P5 threshold ≤ 1.05): b32 1.0478, b8 1.0280
- Ceiling-recovery fraction (baseline−channel)/(baseline−oracle): b32 0.8077, b8 0.7540

## Outcome (declaration §3–§4, applied 2026-10-15)

- **Reproduction gate:** 6 of 210 graph×seed cells differ from the registered caches, max |Δ| 1.24e-05 → drift is negligible; disclose the count.
- **Row 1 applies:** every 8c primary row and P4 keep the same verdict at batch 8 (channel −0.3023, BH-p .0046; spatial −0.3228, .0029; full −0.3006, .0046; P4 −0.1921, p .0077). Wording: "8c/P4 verdicts are unchanged at eval batch 8." O19 → answered for 8c/P4.
- **Placebo rows that change verdict** (disclose beside their primary rows; they do not override the registered verdicts):
  - P4 placebo (scramble − baseline, OTHER): +0.0079, p .028 at b32 → −0.0008, p .89 at b8. The registered OTHER wobble does not appear at batch 8.
  - 8c full-arm placebo stays significant at both batches (−0.0147, p .020 → −0.0149, p .047): already disclosed.
  - 8d OTHER placebo: sign flips, n.s. at both batches (+0.0022, p .71 → −0.0026, p .44). Same-verdict rule flags it on sign only.
- **8d row applies:** oracle − channel CONV becomes significant at batch 8 (−0.0725, p .19 → −0.0986, p .020, MDE 0.098). The paper's 8d paragraph must say the 8d primary is eval-batch-sensitive. Registered batch-32 result (n.s., P5 not confirmed) stays primary. Oracle CONV 1.0478 (b32) / 1.0280 (b8), both ≤ 1.05; recovery 0.81 / 0.75.
- The b8 MDE for the 8d primary (0.098) is just above the observed Δ; do not describe the b8 result as a confirmation of P5.
