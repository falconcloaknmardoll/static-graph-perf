# Power analysis — the paper's §4 rigor-backbone artifact

Tool: `power_analysis.py` (repo root), released alongside the evaluation protocol per
`paper_outline.md` §9 ("Release the protocol, the power-analysis tool, and the
feature/information-gap analysis as a standard").

## 1. Minimum detectable effect (MDE) — formalizes C9

Paired t-test, α=0.05, power=0.8, at the project's own established paired-difference seed
std range (VERIFICATION.md C9: σ≈0.04–0.07, measured across multiple architectural
comparisons):

| σ | n_seeds=3 | n_seeds=5 | n_seeds=7 | n_seeds=10 | n_seeds=15 | n_seeds=20 |
|---|---|---|---|---|---|---|
| 0.040 | 0.124 | 0.067 | 0.051 | 0.040 | 0.031 | 0.026 |
| 0.055 | 0.170 | 0.091 | 0.070 | 0.055 | 0.043 | 0.036 |
| 0.070 | 0.217 | 0.116 | 0.089 | 0.070 | 0.055 | 0.046 |

**Reading:** at the project's standard n_seeds=5, the minimum reliably detectable Δτ is
**0.067–0.116** depending on where the true noise falls in the C9 range. This is a
quantitative validation of C9 from first principles, not just an empirical observation — and
it explains, mechanically, why this project's own results split the way they do:
- The 8c channel-arm effect (Δslowdown=−0.30, later restated on CONV Top-1 slowdown) and the
  xla:default hierarchy gain (Δτ=+0.116, p=.002) both sit **at or above** the 5-seed MDE
  ceiling — which is exactly why both were independently significant.
- Effects below ~0.02–0.05 Δτ (several of the xla:random nulls, Phase 2c/3) are, by this same
  table, **structurally undetectable at n=5** — not evidence of "no effect," but evidence the
  study was underpowered to see one either way. This is the honest reading C9 always implied;
  this table makes it a number instead of an impression.

**Required seeds to detect specific effect sizes** (worst-case σ=0.070):
| target Δτ | n_seeds required |
|---|---|
| 0.02 | 99 |
| 0.05 | 18 |
| 0.10 | 6 |
| 0.15 | 4 |

This is the concrete rebuttal to "isn't 'use more seeds' obvious?" (paper_outline.md's own
anticipated reviewer question) — the field's typical n=1–3 seeds (Phase 7 audit: 20/20 papers
report no seed variance at all) cannot detect anything smaller than a huge effect, and most
architectural claims in this subfield are not huge.

## 2. Tightening curve (τ-std vs #eval-graphs) — status: DONE 2026-08-20

`power_analysis.py tighten` bootstrap-estimates how the seed-to-seed std of the *aggregate* τ
shrinks as the number of evaluation graphs grows — the direct quantification of "you only have
7–20 val graphs, isn't that the point?" Run against the flat (wide-baseline) model's real
per-graph τ on xla:default, from `experiments/phase6b_confound_checks.json`
(`per_graph_tau.wide-baseline`, 7 graphs × 5 seeds):
```
python power_analysis.py tighten \
  --per-graph-json experiments/phase6b_confound_checks.json \
  --key per_graph_tau.wide-baseline \
  --out experiments/power_analysis_tightening.md \
  --figure experiments/figures/power_tightening_curve.png
```
Full table: `experiments/power_analysis_tightening.md`; figure:
`experiments/figures/power_tightening_curve.png`.

**Only 7 graphs are available in xla:default's own validation split** — a real, disclosed
limitation. The bootstrap resamples *with replacement* from those 7, so the curve beyond n=7 is
an extrapolation of what more graphs *would* do under the same per-graph variance, not a claim
that we evaluated on more graphs than we actually did. At n=7 (our real sample size), bootstrap
median std(τ) = 0.0152 (p05–p95: 0.0071–0.0549). Extrapolating to n=20 (the full sweep's target),
median std drops only modestly to 0.0135 (p05–p95: 0.0038–0.0337) — the median is fairly flat
past n≈8, but the **p95 tail shrinks by ~39%** (0.055→0.034), meaning the main benefit of more
eval graphs is capping worst-case unlucky-sample variance, not moving the typical case. Read
together with §1's MDE: at n=7 the achievable resolution is already inside the regime needed to
distinguish the Phase-6b flat-vs-hardware Δτ≈0.165 (≈11× the n=7 median std), so the small-n
val set is not what's limiting that particular claim; it's more relevant to smaller,
closer-to-noise-floor effects.
