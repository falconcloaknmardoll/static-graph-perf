# Phase 8c — pre-registered results (frozen split, paired by seed)

Instances: CONV = 3 graphs × 2 collections, OTHER = 4 × 2.

arm           CONV top-1 slowdown    OTHER top-1 slowdown
baseline         1.4246 ± 0.0721        1.0272 ± 0.0027
channel          1.1203 ± 0.1077        1.0292 ± 0.0182
spatial          1.1758 ± 0.0389        1.0237 ± 0.0080
full             1.1933 ± 0.1479        1.0125 ± 0.0080

## Primary test (§7): CONV subset, arm vs baseline, paired by seed
  channel   Δslowdown=-0.3043  CI95=[-0.3713,-0.2271]  t-p=0.0021  BH-p=0.0032  Wilcoxon-p=0.062  d=-3.15
  spatial   Δslowdown=-0.2488  CI95=[-0.2971,-0.2197]  t-p=0.0005  BH-p=0.0014  Wilcoxon-p=0.062  d=-4.72
  full      Δslowdown=-0.2313  CI95=[-0.3094,-0.1582]  t-p=0.0073  BH-p=0.0073  Wilcoxon-p=0.062  d=-2.25

## Placebo (§7): OTHER subset (must be ~unchanged)
  channel   Δslowdown=+0.0020  CI95=[-0.0124,+0.0164]  t-p=0.8170
  spatial   Δslowdown=-0.0035  CI95=[-0.0108,+0.0040]  t-p=0.4577
  full      Δslowdown=-0.0147  CI95=[-0.0215,-0.0079]  t-p=0.0199

## Decision rules (§8, verbatim)
  gap closes (BH-p<0.05, OTHER unchanged) → positive co-design result
  gap unchanged → representational-floor evidence (publishable)
  OTHER moves too → confound; investigate before any claim
  mixed across arms → report per-arm; +spatial vs +channel contrast IS the finding (P1/P2)

  Registered predictions: P1 channel→null, P2 spatial→best-but-uncertain,
  P3 persisting gap = reportable finding. (Committed 2026-06-29, before data.)

---

## INTERPRETATION & REQUIRED SCRUTINY (added post-hoc, 2026-07-13 — NOT part of the frozen evaluator output)

**Verdict per frozen rules:** channel & spatial → *positive co-design result* (CONV gap closes,
BH-p<0.05, placebo unchanged); full → gap closes but placebo moves slightly (−0.015, p=0.02) →
"investigate." **This REVERSES pre-registered P1** (channel predicted null; was the strongest,
Δ=−0.304). Wilcoxon p=0.062 for all = the n=5 floor (as significant as 5 seeds allow), t/BH significant.

**⚠️ DO NOT CLAIM A CO-DESIGN WIN until these are resolved (the project's verify-before-believe rule;
this result is strong AND contradicts both P1 and our own model-free analysis — max scrutiny):**

1. **Config-invariance puzzle (the crux).** MXU features are derived from each conv node's STATIC
   shape → identical across all configs of a graph. Top-1 slowdown is WITHIN-graph config ranking.
   Config-invariant inputs cannot directly reorder a graph's configs; any effect must be via changed
   training dynamics (model stops ignoring the config signal on conv graphs). **Must demonstrate the
   mechanism, or suspect a bug/leak.** First checks: (a) confirm mxu_block is truly identical across a
   graph's configs in the loaded batches; (b) confirm baseline conv predictions are near-constant
   across configs (a "collapse" the features break).
2. **Specificity control (the Phase-2b lesson).** Is it MXU *information* or just *extra conv-node
   features/capacity*? Add a **scrambled/random conv-feature arm** of matched dim (k=4 to mirror
   channel). MXU > random → real co-design; MXU ≈ random → capacity/regularization effect, reframe.
3. **Reconcile with model-free analysis:** channel-padding vs layout-sensitivity was ρ=−0.06 (n=160),
   yet channel features close the gap. Different quantities (sensitivity ≠ rankability) or one misleads?
4. **full-arm placebo wobble** (−0.015, p=0.02): tiny and improving, but per rules = investigate.

**Framing for the second author / paper:** this is a "surprising positive that reverses our prediction + here are the
two controls that decide if it's real" — the honest, strong version. NOT "we fixed convolution." The
8d oracle probe design should now FOLD IN the random-feature control (item 2) as its first arm.
