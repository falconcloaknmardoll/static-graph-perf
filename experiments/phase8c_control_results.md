# Phase 8c — pre-registered results (frozen split, paired by seed)

Instances: CONV = 3 graphs × 2 collections, OTHER = 4 × 2.

arm           CONV top-1 slowdown    OTHER top-1 slowdown
baseline         1.4246 ± 0.0721        1.0272 ± 0.0027
channel          1.1203 ± 0.1077        1.0292 ± 0.0182
scramble         1.3645 ± 0.0572        1.0351 ± 0.0039

## Primary test (§7): CONV subset, arm vs baseline, paired by seed
  channel   Δslowdown=-0.3043  CI95=[-0.3713,-0.2271]  t-p=0.0021  BH-p=0.0043  Wilcoxon-p=0.062  d=-3.15
  scramble  Δslowdown=-0.0601  CI95=[-0.1454,+0.0285]  t-p=0.3008  BH-p=0.3008  Wilcoxon-p=0.312  d=-0.53

## Placebo (§7): OTHER subset (must be ~unchanged)
  channel   Δslowdown=+0.0020  CI95=[-0.0124,+0.0164]  t-p=0.8170
  scramble  Δslowdown=+0.0079  CI95=[+0.0038,+0.0121]  t-p=0.0283

## Decision rules (§8, verbatim)
  gap closes (BH-p<0.05, OTHER unchanged) → positive co-design result
  gap unchanged → representational-floor evidence (publishable)
  OTHER moves too → confound; investigate before any claim
  mixed across arms → report per-arm; +spatial vs +channel contrast IS the finding (P1/P2)

  Registered predictions: P1 channel→null, P2 spatial→best-but-uncertain,
  P3 persisting gap = reportable finding. (Committed 2026-06-29, before data.)

## Specificity control (prereg A2): CONV subset, channel vs scramble, paired by seed
  P4: channel significantly < scramble ⇒ real MXU info; channel ≈ scramble ⇒ capacity artifact.
  channel−scramble  Δslowdown=-0.2442  CI95=[-0.3154,-0.1606]  t-p=0.0063  Wilcoxon-p=0.062  d=-2.35
  placebo (scramble−baseline, OTHER)  Δ=+0.0079  t-p=0.0283
