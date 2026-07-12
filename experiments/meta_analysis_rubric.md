# Meta-Analysis Rubric — Auditing Evaluation Rigor in Learned Cost-Model Papers

**Purpose.** A formal, reproducible instrument for §6 of the paper. Audit ~30 recent
"ML for Compilers" / "learned cost model" / "GNN for program performance" papers and score
each on five axes. The goal is a *fair, pre-registered* categorization — not cherry-picking
villains. Decide the rubric **before** reading the papers; apply it identically to all.

**How to use.**
- Build a spreadsheet: one row per paper, one column per axis (P / ∂ / F), plus `notes`,
  `venue`, `year`, `task`, `our-conclusion-affected?`.
- Score **P (Pass) / ∂ (Partial) / F (Fail)** per axis using the explicit anchors below.
- When in doubt, score the *weaker* category and justify in `notes` (conservative = credible).
- Pilot on 5 papers first; if two raters disagree, sharpen the anchor, don't fudge the score.
- Headline statistics to report: % Fail per axis, % of papers failing ≥3 axes, and the
  fraction whose *headline architectural claim* rests on an uncontrolled comparison.

> Scope rule: only audit claims of the form "architecture/feature X improves ranking/accuracy
> over baseline Y." Papers making no comparative architectural claim are out of scope (record
> as N/A), so the audit targets exactly the claim type our results undercut.

---

## Axis 1 — Variance & Seeds
**Question:** Is the reported result a distribution over multiple random seeds, or a single
point estimate?
- **P** — ≥3 seeds (ideally ≥5), reports mean **and** dispersion (std / CI / error bars).
- **∂** — multiple runs but no dispersion reported, OR "best of N" only, OR seeds for some
  results but not the headline comparison.
- **F** — single run / unstated seeds for the headline number.

**Why it matters:** our measured seed std (≈0.04–0.07) exceeds typical reported gains; a
single point estimate cannot distinguish signal from seed luck. *Record:* #seeds, dispersion
type, whether the headline τ/accuracy has error bars.

## Axis 2 — Parameter / Compute-Matched Comparison
**Question:** Is the proposed model controlled for capacity against its baselines, so a win
is attributable to the *idea* rather than to more parameters/FLOPs/training?
- **P** — explicit parameter or FLOP match (within a stated tolerance), or a capacity
  sweep showing the gain is not just scale.
- **∂** — similar magnitude but no explicit match; or matches params but not batch
  size / training budget / data seen.
- **F** — proposed model is larger/longer-trained than the baseline with no control (the
  exact confound that produced our Phase 2b artifact).

**Why it matters:** our DiffPool "win" (235k vs 136k params) vanished under a parameter
match. *Record:* param counts of model vs baseline, whether batch/training were also matched.

## Axis 3 — Statistical Significance Testing
**Question:** Is the claimed improvement supported by a hypothesis test, or asserted from a
raw number being larger?
- **P** — appropriate test (paired t / Wilcoxon / bootstrap CI) with reported statistic,
  matched-seed pairing where applicable.
- **∂** — reports variance but no test, leaving "is the gap > noise?" to the reader's eye.
- **F** — "ours = 0.36 vs theirs = 0.34, therefore better" with no variance or test.

**Why it matters:** with overlapping ±0.04 bands, "higher mean" is not "better." *Record:*
test used (if any), effect size, p-value, whether comparison was paired.

## Axis 4 — Evaluation-Set Integrity
**Question:** Is the headline number on a held-out **test** set of adequate size, or on the
same validation graphs used for selection/tuning?
- **P** — official/held-out test set, adequate graph count, selection done on a separate
  split; tie/normalization protocol stated.
- **∂** — validation-only but honestly labeled, or test set with very few graphs (single
  digits) and no power caveat.
- **F** — reports the metric on the set used to pick the model/hyperparameters (selection
  leakage), or never states which split.

**Why it matters:** 7–20 evaluation graphs is high-variance; selection-set reporting inflates
results. *Record:* split used, #eval graphs, whether selection and reporting splits differ.

## Axis 5 — Effect Size vs Noise & Baseline Strength
**Question:** Is the claimed improvement large relative to plausible noise, and is the
baseline a *strong, well-tuned* model rather than a strawman?
- **P** — gain clearly exceeds reported variance **and** the baseline is competitive
  (tuned, capacity-matched, the actual prior SOTA).
- **∂** — gain comparable to its own error bars, or baseline is reasonable but lightly tuned.
- **F** — sub-noise gain (e.g. <0.02 τ with unknown variance), or baseline is an
  undertuned strawman that inflates the apparent improvement.

**Why it matters:** a real contribution needs an effect bigger than the noise floor *and* a
baseline worth beating. *Record:* claimed Δ, its ratio to reported/estimated noise, how the
baseline was tuned.

---

## Aggregation & reporting
- **Per-axis fail rate:** `% F` and `% (∂ or F)` across the corpus — the headline numbers.
- **Compound failure:** `% of papers failing ≥3 of 5 axes` — the "statistically unsupported
  headline" rate.
- **Claim-level:** of papers whose central claim is architectural, what fraction would *not
  survive* a parameter-matched multi-seed paired test (Axes 1–3 all P)? This sentence is the
  paper's money quote.
- **Fairness controls:** pre-registered rubric, two independent raters on a ≥20% sample with
  reported agreement (Cohen's κ), conservative tie-breaking, and inclusion of strong
  counter-examples (papers that *do* evaluate rigorously) so the audit reads as calibrated,
  not adversarial.
