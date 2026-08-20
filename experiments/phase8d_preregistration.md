# Phase 8d — Pre-registration (oracle/upper-bound probe, candidate 1)

> **Status: FROZEN 2026-08-19, before running 8d.** Mirrors `phase8c_preregistration.md`'s
> discipline: hypotheses, split, arm, metric, test, and decision rules fixed in advance.
> Any deviation is a dated amendment below, not a silent edit. This experiment is now
> **optional capstone, not load-bearing** — Phase 8c's confound-A specificity control already
> passed on its own (`experiments/phase8c_results.md`), so the paper's L2 claim does not depend
> on this result either way. It remains valuable as the direct answer to the one open reviewer
> hole flagged in `paper_outline.md`: *"conv is just hard, not unrepresentable."*

## 1. Motivation (what we already know, with evidence)
- **8c confirmed (2026-08-19):** encoding static MXU geometry (`channel` arm) significantly
  closes the convolution Top-1 slowdown gap — 1.4246→1.1203 (BH-p=.004) — and a specificity
  control confirms this is real hardware information, not a capacity artifact (channel
  significantly beats a scrambled-feature control, Δ−0.244, p=.006).
- **But the gap does not close to 1.0.** Residual CONV Top-1 slowdown of 1.12 means the best
  static-feature arm still picks a config that runs ~12% slower than optimal, on average. Two
  readings are consistent with this residual: (a) 1.12 is close to the achievable ceiling given
  ONLY static shape information — the remaining gap is a genuinely hard, near-irreducible
  residual; or (b) there is substantial headroom still recoverable by ANY informative
  per-config signal, meaning the floor is a **representation** problem (the model still lacks
  config-specific information), not a fundamental limit of the task.
- **This experiment distinguishes (a) from (b) directly** by giving the model a deliberately
  leaky, non-deployable oracle signal and measuring how far CONV Top-1 slowdown can be pushed —
  establishing the ceiling channel/8c's static features are being measured against.

## 2. Research question
If the model is given a low-dimensional, deliberately-leaky per-config oracle signal derived
from that config's own true runtime, does CONV-subset Top-1 slowdown recover toward 1.0 (proving
the post-8c residual is representational, i.e. more per-config information would still help), or
does it plateau well above 1.0 even with a leaky signal (proving the residual is closer to a
genuine floor — label noise / runtime non-determinism / a harder ceiling than expected)?

## 3. Frozen hypotheses & registered predictions
- **H1 (primary):** `baseline + oracle-bucket` reduces CONV Top-1 slowdown well below the
  `channel` arm's 1.1203, plausibly toward the 1.0–1.05 range, paired across seeds vs both
  `baseline` and `channel`. Null H0: no further improvement over `channel`.
- **Registered prediction P5 (committed now, before the run):** the oracle arm will recover
  CONV Top-1 slowdown to **≤1.05** (a stronger claim than "significantly better than channel" —
  a specific, falsifiable ceiling estimate). If it does NOT clear 1.05, the residual floor is
  sharper/harder than the "just needs more information" story suggests, which is itself a
  reportable, more interesting finding (points at label noise/non-determinism, the C10 open
  question in `feedback-verification-pass`).
- **Ceiling-recovery framing (secondary, descriptive not hypothesis-tested):** report
  `(baseline − channel) / (baseline − oracle)` as the fraction of the achievable ceiling that
  8c's static MXU features already recovered — this is the number that actually answers
  "how close did 8c get," independent of whether P5 itself is cleared.

## 4. The oracle signal (frozen design, decided BEFORE running)
**Per-config rank-bucket from the graph's OWN true runtime distribution.** For each graph, sort
all configs by true `config_runtime`, assign each config to one of **K=5 quantile buckets**
(fastest 20% = bucket 0, ..., slowest 20% = bucket 4), append the bucket as a **one-hot 5-dim
column block** to the existing 158-dim Early-Join input (158→163), computed **within-split**
(train configs bucketed against train-split runtimes for that graph; validation configs bucketed
against the validation-split runtimes for that graph — no train/val leakage of the SPLIT
boundary itself, only of the label ordering *within* whichever split a config already belongs
to, which is the whole and only point of an oracle probe).
- **K=5 is frozen now, not tuned post-hoc.** Coarser than a literal rank (which would trivially
  solve the task and prove nothing) but fine enough to be strongly informative — a deliberate,
  disclosed middle ground between "no leak" (channel) and "total leak" (an oracle giving the
  literal answer, which is not an interesting ceiling because it doesn't correspond to any
  physically realizable amount of compiler-derivable information).
- **This is an oracle, never a deployable feature.** True runtime is not knowable at inference
  time in any real deployment; every reference to this arm in the paper must say so explicitly.
  It exists solely to calibrate a ceiling, matching `phase8d_oracle_probe.md`'s own framing.

## 5. FROZEN data split (reused from 8c, not re-derived)
Identical to `phase8c_preregistration.md` §4: CONV subset = `inception_v3_batch_128_train`,
`resnet50.4x4.fp16`, `resnet_v1_50_official_batch_128_bf16`; OTHER subset = the 4 bert/unet
graphs. Evaluated on both `xla/default/valid` and `xla/random/valid`. Train on
`layout:xla:random`, matching 8c's protocol exactly (Amendment A1).

## 6. Arms (fixed set, run all)
1. **baseline** — 158-dim, no MXU features, no oracle (reused checkpoint from 8c, unchanged).
2. **channel** — 158+4 MXU features (reused checkpoint from 8c, unchanged) — the arm being
   calibrated against.
3. **oracle-bucket** — 158+5 rank-bucket one-hot (NEW, 5 seeds to train).
(No BH correction needed — single new arm, two pre-specified comparisons: oracle vs baseline,
oracle vs channel.)

## 7. Training protocol (matched to 8c — no changes)
Wide flat baseline (`--model wide-baseline`, capacity grows with input dim per 8c's own
precedent), `--configs-per-batch 8`, `--max-configs-per-graph 128`, cosine LR 1e-3→1e-5,
grad-clip 0.1, dropout 0, **5 seeds {42,100,123,456,789}**, paired by seed, capped harness
(`launch_capped.sh`, num-workers 0). ⛔ GPU still down as of this writing — same CPU workaround
as 8c's scramble control; budget similarly (8c's scramble arm took ~8hr wall/seed-pair on CPU
for evaluation alone, training was the ~10hr/seed figure from the 2026-08-03 timing test).

## 8. Primary metric & statistical test
- **Primary:** CONV-subset Top-1 slowdown, `oracle-bucket` vs `channel`, paired across 5 seeds —
  same paired t + Wilcoxon + bootstrap 95% CI as 8c §7.
- **Secondary:** `oracle-bucket` vs `baseline` (magnitude of the full achievable gain).
- **Placebo:** OTHER subset for `oracle-bucket` vs `baseline` — note this placebo is
  **expected to show SOME movement**, unlike 8c's (the bucket signal is genuinely informative
  everywhere, not conv-specific by construction) — report it descriptively, do not require it
  to be null; the discipline check here is instead that the CONV-subset gain is **not smaller**
  than the OTHER-subset gain (if OTHER improves MORE than CONV, that undercuts the
  conv-specific-floor story and must be reported honestly).
- **Significance threshold:** p<0.05, oracle vs channel, CONV subset.

## 9. Decision rules (fixed before data)
| Outcome | Interpretation (committed) |
|---|---|
| Oracle CONV slowdown ≤1.05, significantly beats channel | **P5 confirmed** — the post-8c residual is representational; static MXU features recovered `(baseline−channel)/(baseline−oracle)` of the achievable ceiling. Specifies what a faithful microarch model must still compute. |
| Oracle beats channel but stays well above 1.05 | **Partial** — some headroom exists but the ceiling is lower than hoped; report the actual number, do not round up to "solved." |
| Oracle ≈ channel (no significant further gain) | **Sharper finding than expected** — even a leaky per-config oracle can't move the needle further, pointing at label noise / runtime non-determinism (C10) as the deeper limit, not missing representation. Publishable as a more surprising result. |
| OTHER-subset gain ≥ CONV-subset gain | **Confound-adjacent** — the bucket signal is not conv-specific; report honestly, reframe the "floor" claim as an information-content ceiling in general, not a conv-specific one. |

## 10. Anti-p-hacking commitments
- K=5 buckets is frozen now; will not be re-tuned (e.g. to K=10) after seeing a disappointing
  result.
- The ≤1.05 threshold in P5 is a specific, falsifiable number committed before data, not a
  post-hoc "significantly better" claim with no magnitude attached.
- 5 seeds fixed; no seed-adding to chase significance, matching 8c's own commitment.
- This is explicitly framed as optional/capstone in §0 above — there is no pressure to make it
  land, since the paper's core L2 claim does not depend on the outcome either way.
