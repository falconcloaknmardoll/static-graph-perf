# Phase 8d — pre-registered oracle capstone results (frozen split, paired by seed)

Instances: CONV = 3 graphs x 2 collections, OTHER = 4 x 2. baseline/channel reused
unchanged from checkpoints/phase8c/eval_cache/ (no new inference for those two arms).

arm               CONV top-1 slowdown    OTHER top-1 slowdown
baseline             1.4246 ± 0.0721        1.0272 ± 0.0027
channel              1.1203 ± 0.1077        1.0292 ± 0.0182
oracle-bucket        1.0478 ± 0.0064        1.0294 ± 0.0146

## Primary test (prereg §8): oracle-bucket vs channel, CONV subset, paired by seed
  oracle-bucket vs channel  Δslowdown=-0.0725  CI95=[-0.1501,+0.0052]  t-p=0.1903  Wilcoxon-p=0.312  d=-0.70

## Secondary: oracle-bucket vs baseline, CONV subset (full achievable gain)
  oracle-bucket vs baseline  Δslowdown=-0.3768  CI95=[-0.4308,-0.3222]  t-p=0.0003  Wilcoxon-p=0.062  d=-5.37

## Placebo (prereg §8): OTHER subset, oracle-bucket vs baseline
  NOT required to be null (bucket signal is informative everywhere by construction).
  Discipline check: CONV gain must be >= OTHER gain, or the conv-specific-floor
  story is undercut and must be reported honestly.
  oracle-bucket vs baseline (OTHER)  Δslowdown=+0.0022  CI95=[-0.0086,+0.0106]  t-p=0.7055
  CONV gain=+0.3768  OTHER gain=-0.0022  CONV>=OTHER: True

## Registered prediction P5 (committed 2026-08-19, before data)
  oracle-bucket CONV Top-1 slowdown = 1.0478   threshold <=1.05   PASS=True
  Ceiling-recovery fraction (baseline-channel)/(baseline-oracle) = 0.8077

## Decision rules (prereg §9, verbatim)
  oracle<=1.05 & sig. beats channel -> P5 confirmed: post-8c residual is representational
  oracle beats channel but >1.05    -> partial: headroom exists, ceiling lower than hoped
  oracle ~= channel (no sig. gain)  -> sharper finding: residual is a floor (label noise /
                                        non-determinism, C10), not missing representation
  OTHER gain >= CONV gain           -> confound-adjacent: bucket signal not conv-specific
