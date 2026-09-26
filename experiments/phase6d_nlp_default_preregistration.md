# Phase 6d — Pre-registration: the nlp:default prediction

> **Registered 2026-09-22, BEFORE any `nlp/default` validation τ was inspected.**
> Drafted by Claude at the first author's instruction; the first author is the registering author. Frozen on commit.
> Conventions follow `phase8c_preregistration.md`. Standing rule
> (`project-prereg-rigor-framing`): a pre-registered result is never softened post-hoc, and
> **"was it powered?" is asked before "how do I phrase this?"** — §6 is therefore written
> *before* §7, and §7 is calibrated to what §6 says is reachable.

## 0. Observation status — the anti-HARKing artifact

State of `checkpoints/expansion/nlp_default/` at registration time (verified 2026-09-22):
- **0 of 10 runs carry a `.done` marker.**
- **9 of 10 runs have not started.**
- **1 run in progress**: `wide-baseline-seed42`, at epoch 39/50 (~780 s/epoch).
- **No `val_tau` has been read for any `nlp_default` run** by the first author or by Claude. The only
  `nlp_default` output either party has viewed is training-**loss** lines from a process
  check (`loss=1.221…`), which is not the outcome measure.

The outcome measure (§5) is therefore **unobserved** for ≥9.8/10 of the design. This document
is registered against genuinely unseen data.

## 1. Motivation — what is already known (with evidence)

Three of four cells of a 2×2 (`{xla, nlp}` × `{default, random}`) are measured. The primary
contrast throughout is **paired `hardware` − `wide-baseline`** (param-matched flat model,
h=172, ~236k params), 5 matched seeds `{42,100,123,456,789}`.

| Collection | paired Δτ (hw − flat) | p | Reading |
|---|---|---|---|
| **xla:default** | **+0.1650 ± 0.0133** | **p < 0.001** | hardware wins decisively (~12× paired seed-σ) |
| **nlp:random** | **−0.0722 ± 0.0583** | **p = 0.0504** | hardware is *marginally **worse*** — **not a clean null** |
| xla:random | ≈ null | see §9 discrepancy | no significant benefit |
| **nlp:default** | **← THIS PREREGISTRATION** | — | unobserved |

**The structure this suggests:** the benefit of hierarchy + hardware-contention edges is
**distribution-dependent** — present on `default` (the compiler's own preferred-layout
distribution), absent-or-negative on `random`. If that axis is real and not an `xla`-specific
accident, `nlp:default` should pattern with `xla:default`.

## 2. Research question

**Does the hardware/hierarchy advantage track the `default` vs `random` configuration
distribution, independently of the model family (`xla` vs `nlp`)?**

## 3. Frozen hypotheses & registered predictions

- **P1 (primary, directional).** On `nlp:default`, paired Δτ (hardware − wide-baseline) is
  **positive and significant** at α = 0.05, two-sided.
- **P2 (secondary, pattern).** The sign of Δτ on `nlp:default` matches `xla:default`
  (positive) and differs from `nlp:random` (negative) — i.e. the split is by *distribution*,
  not by *model family*.
- **P3 (magnitude, exploratory — NOT a pass/fail criterion).** If P1 holds, Δτ on
  `nlp:default` is of the same order as `xla:default` (+0.1650). Registered as exploratory
  **because §6 shows n=5 cannot adjudicate "same order" with any useful precision.** It is
  recorded to prevent a post-hoc magnitude claim, not to be tested.

**Registered alternative, stated in advance so a null is not spun:** if Δτ on `nlp:default` is
null or negative, the distribution-dependence axis is **not** general, and Result I must be
reported as an **`xla:default`-specific** finding rather than a `default`-vs-`random` law.
This is the outcome that most damages the current framing, and it is registered as such.

## 4. Data, arms, protocol — already fixed, no degrees of freedom

Fixed by `run_collection_expansion.sh` before registration; **nothing below may be changed**:
- **Collection:** `data/predict-ai-model-runtime/npz_all/npz/layout/nlp/default`
- **Arms:** `wide-baseline` (param-matched flat) vs `hardware`. 2 arms × 5 seeds = 10 runs.
- **Protocol:** identical to Phase 2c/3/6 — 50 epochs, `--configs-per-batch 8`,
  `--max-configs-per-graph 128`, `--eval-batch 8`, `--dropout 0.0`, `--lr 1e-3`,
  `--weight-decay 1e-4`, `--grad-clip 0.1`; `hardware` additionally carries `--lp-weight` /
  `--entropy-weight`.
- **Split:** the dataset's own `valid/` split. Not re-derived, not re-tuned.
- **Selection:** `best_model.pt` → `val_tau`, exactly as the other three cells were scored.

## 5. Primary metric & test

- **Metric:** Kendall's τ on the held-out `valid/` split; **paired per-seed differences**
  `Δ_s = τ_hardware(s) − τ_flat(s)`.
- **Primary test:** paired two-sided t-test, α = 0.05, n = 5.
- **Reported alongside (not gates):** Wilcoxon signed-rank (n=5 floor is p=0.0625 — it
  **cannot** reach 0.05, so it is reported for direction only, never as a pass criterion),
  bootstrap CI on mean Δτ, Cohen's d, and the per-seed table.
- **Multiplicity:** P1 is a single pre-specified primary contrast — no correction. P2 is
  descriptive. P3 is exploratory.

## 6. POWER — computed BEFORE the decision rules (the Phase 8d lesson)

Phase 8d registered a hypothesis whose two conditions were **mutually unsatisfiable**: its
ceiling capped the effect below the n=5 MDE, so it could not pass however the data fell. That
failure mode is checked for here *first*.

Best available σ prior for an `nlp` collection = `nlp:random`'s paired σ = **0.0583**
(`xla:default`'s 0.0133 is not used — it would flatter the design).

Minimum detectable effect, paired t, α=0.05, power=0.8, n=5 (`power_analysis.py mde`):

| σ (paired) | MDE (Δτ) |
|---|---|
| 0.0133 (`xla:default`) | 0.0221 |
| **0.0583 (`nlp:random` — the assumed prior)** | **≈ 0.097** |
| 0.0700 (worst case) | 0.1164 |

**Satisfiability check — PASSES.** The registered prediction P1 is reachable: an effect the
size of `xla:default`'s (**+0.1650**) exceeds the MDE at the assumed σ (**0.097**) and even at
worst-case σ (**0.1164**). Unlike 8d, **no ceiling in this design caps the effect below the
MDE.** P1 can be confirmed by the data if it is true.

**⚠️ The asymmetry, registered in advance.** This design is powered to *detect* an
`xla:default`-sized effect, **not** to establish absence. At n=5, σ≈0.058:
- A **half-sized** effect (Δτ ≈ +0.08) would likely be **missed** (below MDE).
- Therefore a non-significant result **cannot** be reported as "nlp:default is null." The only
  admissible null statement is: *"no effect larger than ≈0.10 τ was detected at n=5."*
- **Registered now:** if P1 fails, the write-up **must** quote the realized MDE next to the
  null. Any phrasing implying a demonstrated absence is a violation of this registration.

## 7. Decision rules (fixed before data; asymmetric by design)

| Outcome | Verdict | Required reporting |
|---|---|---|
| Δτ > 0, paired p < 0.05 | **P1 CONFIRMED.** Distribution-dependence supported as a general axis; the 2×2 completes. | Report Δτ ± σ, p, CI, d, per-seed table. Do **not** claim magnitude equality with `xla:default` (P3 is exploratory). |
| Δτ > 0, p ≥ 0.05 | **P1 NOT CONFIRMED** — directionally consistent, underpowered. | Report as directional-only. Quote realized MDE. **No "trending toward significance."** |
| Δτ ≈ 0, p ≥ 0.05 | **P1 FAILS.** | Report Result I as **`xla:default`-specific**, per §3's registered alternative. Quote MDE; do not claim absence. |
| **Δτ < 0** (any p) | **P1 FAILS and P2 FAILS.** | Report in full. `nlp:default` patterning with `nlp:random` would mean the axis is **model family, not distribution** — a materially different and publishable finding. **Registered in advance so it cannot be framed as a wash.** |

**No outcome licenses dropping `nlp:default` from the paper.** It is reported whichever way it
falls.

## 8. Anti-p-hacking commitments

1. **n = 5 seeds, fixed.** No seed may be added after seeing results; no seed dropped as an
   outlier without reporting both the with- and without- analyses.
2. **No metric substitution.** Kendall's τ on `valid/` is the outcome. Top-K, MSE and slowdown
   are not fallbacks if τ disappoints.
3. **No arm substitution.** `hardware` vs `wide-baseline` is the contrast. The Phase 6b
   `plain-DiffPool` decomposition arm was **not run** for this collection and may not be added
   post hoc to rescue a null.
4. **Best-checkpoint rule unchanged** — `best_model.pt` / `val_tau`, as for the other 3 cells.
5. **The registered alternative in §3 is binding.** A P1 failure is written up as an
   `xla:default`-specific finding, not as an inconclusive aside.
6. **Amendments** append below with a date and must be added **before** the relevant data is
   inspected.

## 9. ⚠️ Discrepancy to resolve BEFORE submission (not part of this registration)

`experiments/paper_outline.md` §5 (Result I) cites, for "xla:random / nlp", *"paired p=.22/.72"*
— i.e. it describes the nlp cell as a **clean null**.

The Phase 6 expansion checkpoints for **`nlp_random`** (10/10 `.done`, read 2026-09-22) give:

```
wide-baseline τ = +0.7129 ± 0.0577
hardware      τ = +0.6408 ± 0.0334
paired Δτ       = −0.0722 ± 0.0583     paired t p = 0.0504
```

That is a **marginally significant negative** effect, not p=.72.

### ✅ RESOLVED 2026-09-22 — traced, no number was wrong

Both outline p-values are real and correctly sourced. **Neither is the nlp cell:**

| Value | Source | What it actually measures |
|---|---|---|
| **p = .22** | `phase2c_ablation_results.md:94` (Δτ=−0.0117, t=−1.45) | **hierarchy** (DiffPool) vs matched flat, on **layout:xla:random**, 7 val graphs |
| **p = .72** | `phase3_results.md:68` (Δτ=−0.0116, t=−0.39) | **hardware-contention edges** vs matched flat, on **layout:xla:random** |

Confirmed by `phase4_topk_results.md:70-71`, which tabulates them as *"2c — Does hierarchy
help? p=0.22"* and *"3 — Do hardware edges help? p=0.72"*.

**Diagnosis: attribution ambiguity, not a fabricated statistic.** `.22` and `.72` are **two
interventions on ONE collection (xla:random)** — not two collections. The outline placed two
slashes in parallel (`xla:random / nlp` … `p=.22/.72`), which invites a one-to-one mapping of
`.72` onto nlp. It does not belong to nlp.

**The substantive consequence stands:** the outline asserted the nlp cell was null **with no
supporting statistic**, and the real measurement (`nlp_random`, Phase 6 sweep, 10/10 `.done`)
is a **marginally significant negative**, p=.0504 — hardware *hurts*. "Not significantly
better" is the wrong description of that result.

**Fixed 2026-09-22** in `paper_outline.md` §"Finding 1" and §5 (fact bullets only, per the
manuscript-authorship boundary): the two collections are now split into separate bullets, each
p-value is attributed to its own intervention and source file, and the nlp:random numbers are
stated explicitly. **Note this reading strengthens the distribution-dependence thesis** —
`default` positive, `random` null-to-negative — but it must be reported as a negative, not
laundered into a null.

---

## 10. AMENDMENT 1 — 2026-09-25 (per §8.6): registered-arm training failure + amended protocol

> **Status: DRAFT until committed.** Drafted by Claude at the first author's instruction; the first author is the
> registering author. Written **before** the τ of repro run B (§10.3) or any amended-protocol
> run (§10.5) was read. The eval producing run B's τ was running when this was written, and
> its output was deliberately not opened.

### 10.1 Disclosure — §0's "unobserved" statement no longer holds

On 2026-09-24/25, while diagnosing a flat training loss, Claude read (and reported to the first author)
the following `nlp_default` validation τ values. **This is a breach of §0's premise and is
disclosed in full:**

| Run | What was read | Values |
|---|---|---|
| `hardware-seed42` (`.done`) | every eval | **+0.0000** at all 10 evals; best +0.0000 @ epoch 5 |
| `hardware-seed100` (stopped at epoch ~35) | evals at epochs 5–30 | **+0.0000** at all 6 |
| `wide-baseline-seed42` (`.done`) | every eval | rising to best **+0.2543** @ epoch 50 |
| `wide-baseline-seed100` (`.done`) | evals at epochs 20, 40 + wandb sparkline | +0.1673 (best +0.2041 @ 15); +0.3361 (best +0.3369 @ 35) |
| repro run C (§10.3), not a sweep run | final checkpoint | **+0.2760**, 20/20 graphs |

An implied paired Δτ of roughly −0.25 was mentioned in conversation. It is **not** a
registered result (see §10.4). Seeds 123/456/789 of both arms were never run and remain
unobserved.

### 10.2 The failure — the registered `hardware` arm does not train on `nlp:default`

Under the frozen §4 protocol, the `hardware` arm collapses to a **constant output** early in
epoch 1:
- The ListMLE term is pinned at **1.3256**, the constant-score value, from epoch 1 onward
  (both seeds).
- The `hardware-seed42` best checkpoint has **79.9%** of its parameters at |w| < 1e-30,
  score std exactly 0 on every batch, and zero gradient in every layer. It cannot recover.
- **Why the logged τ is exactly 0.0000:** constant predictions make Kendall's τ NaN on every
  graph. `train.evaluate()` drops NaN graphs and returns 0.0 when none remain
  (`val_n_graphs = 0`). **These zeros are not measured correlations. They mean the metric is
  undefined.**
- `wide-baseline` trains normally on the same data under the same protocol.

### 10.3 Diagnosis (`scripts/repro_hardware_collapse.py`, GPU, same loop as `train.py`)

The repro reproduces the collapse deterministically. With the frozen recipe (lr 1e-3, lp 0.1,
ent 0.01), the ranking loss pins by step ~240 (seed 100) and ~540 (seed 42), and weight
death begins at step ~1160 (both seeds). Seed-100 interventions, identical batches,
1,500 steps:

| Run | lr | lp / ent | Outcome |
|---|---|---|---|
| A | 1e-3 | 0 / 0 | **collapses.** The auxiliary DiffPool losses are **not** the cause. |
| B | 1e-4 | 0.1 / 0.01 | scores stay non-degenerate through step 1500. τ **not read** at time of writing. |
| C | 1e-4 | 0 / 0 | non-degenerate; **τ = +0.2760** (20/20 graphs) at step 1499 |

**Diagnosis: the learning rate is the operative factor.** Still unexplained: a slow rise in
near-zero parameters from step ~1200 (0.8% by step 1500), seen at both learning rates.

### 10.4 Registered outcome for the frozen protocol (binding)

- **The §4 protocol cannot produce a valid `hardware` arm on `nlp:default`.** The registered
  result is a **training failure**, and it is reported as such.
- The §7 decision table is **not applied** to it. The Δτ < 0 row in particular ("model
  family, not distribution") **must not** be invoked on an arm whose metric is undefined.
- P1, P2 and P3 are **untested under the registered protocol**. They are not failed and not
  confirmed.
- **Reported in the paper regardless of §10.5:** the shared recipe trains the `hardware`
  model on `xla:default` but not on `nlp:default`. This is a finding about trainability, not
  about information content. The diagnostics above go in the appendix.

### 10.5 Amended secondary protocol (AMENDED, not confirmatory)

The amended learning rate was chosen **after** seeing diagnostic runs, one of which had a τ
readout (run C). Results under this protocol are therefore labelled **amended/exploratory**
everywhere they appear, and they never replace §10.4.

- **Protocol:** identical to §4 except **`--lr 1e-4` for BOTH arms**. The cosine floor
  scales automatically to 1e-6. Same seeds {42,100,123,456,789}, same auxiliary weights, same
  eval cadence (`--eval-every 5`, unchanged, because it determines best-checkpoint
  selection), same `best_model.pt` rule.
- **All 10 runs are fresh**, written to a new directory
  (`checkpoints/expansion/nlp_default_lr1e-4/`). No lr-1e-3 checkpoint is reused in the
  primary amended contrast.
- **Primary amended contrast (A-P1):** paired Δτ (hardware@1e-4 − flat@1e-4) > 0,
  two-sided paired t, α = 0.05, n = 5. Power per §6 is unchanged (MDE ≈ 0.097 at σ = 0.058).
- **⚠️ Confound check (required for A-P1 to be stated as supported):** the LR was chosen to
  rescue the hardware arm, and lowering it may **handicap the flat arm**, which already trains
  at 1e-3. This would inflate Δτ. A-P1 is reported as supported **only if** hardware@1e-4 also
  beats **flat@1e-3** (paired, same test). That requires completing flat@1e-3 seeds
  123/456/789 (3 additional runs, same as the §4 flat arm). If A-P1 passes but this check
  fails, the verdict is **"not supported — attributable to the flat arm's LR"**.
- **Collapse rule:** any amended run whose final `evaluate()` returns `val_n_graphs = 0` is
  entered as **τ = 0** in its test (conservative for the hardware arm), flagged, and reported.
  No seed is replaced.
- §5, §6, §7 (read as A-P1 / A-P2) and §8's commitments otherwise apply unchanged, with every
  reported number labelled **amended**.
- **Feasibility is not a scientific criterion.** If the amended runs cannot finish before
  submission, only §10.4 is reported. No partial-seed amended result is reported as a test.
