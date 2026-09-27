# Phase 6e — Pre-registration: is the xla:default win robust to tuning the flat baseline's LR?

> **Registered 2026-09-27, BEFORE any flat run at a non-1e-3 LR existed.** Drafted by Claude
> at the first author's instruction; the first author is the registering author and approved the design. **Frozen on
> commit and pushed right away** so the timestamp is on the server. No run in this design may
> start before that commit.
> Conventions follow `phase6d_nlp_default_preregistration.md` and `phase8c_preregistration.md`.
> Standing rule (`project-prereg-rigor-framing`): a pre-registered result is never softened
> after the fact, and **"was it powered?" is asked before "how do I phrase this?"** (§6 comes
> before §7). Answers objection **O5** (as rewritten) and bears on **O2**, both in
> `DEFENSIBILITY.md`.

## 0. Observation status — the anti-HARKing artifact

At registration time (verified 2026-09-27):
- **No flat (`wide-baseline`) run on `xla:default` exists at any LR other than 1e-3.** The
  directory `checkpoints/flatlr/` does not exist. The outcome measure for every new arm in this
  design is therefore **unobserved**.
- **Already known, and used below as motivation (not as outcome data):** the Phase 6/6b results
  at lr 1e-3 (§1), and the training-loss trace of `xla_default_wide-baseline_seed42.log`.
- **Running at registration time:** O1's eval-batch re-scoring
  (`scripts/eval_batch_sensitivity.py`) of the *existing* lr-1e-3 checkpoints. It re-scores
  known runs and produces no data about any LR in this design.

## 1. Motivation — what is already known (with evidence)

Result I's `xla:default` cell, 5 matched seeds `{42,100,123,456,789}`, lr 1e-3 for all arms
(`checkpoints/expansion/xla_default/phase6b_summary.txt`):

| Arm | val τ (mean ± sd) | per seed |
|---|---|---|
| flat (`wide-baseline`, h=172, ~236k params) | **0.0327 ± 0.0135** | .027 .031 .021 .028 .056 |
| plain DiffPool (`hierarchical`) | 0.1484 ± 0.0276 | .174 .120 .181 .129 .137 |
| hardware DiffPool (`hardware`) | 0.1977 ± 0.0133 | .193 .212 .196 .179 .209 |

Registered contrasts: hardware − flat **+0.1650 ± 0.0133, p < 0.001**; hierarchical − flat
**+0.1157, p = 0.0019**; hardware − hierarchical +0.0493, p = 0.0299.

**Why this check is needed (objection O5, rewritten 2026-09-27).**
1. **The flat arm is far below the published baseline.** TPUGraphs (Phothilimthana et al.,
   `papers/2308.13490v3.pdf`, Table 2) reports a best-baseline **validation τ = 0.12** on
   `Layout:XLA:Default`. Our flat arm reaches **0.033**.
2. **The flat arm barely trained.** In the seed-42 log, the ListMLE loss moved only
   1.304 → 1.291 across 50 epochs, and its best τ (+0.0269) came at epoch 25.
3. **A competing explanation exists.** One shared LR (1e-3) is *matched*, not *equally tuned*
   (O2). The same shared LR makes the hardware arm collapse on `nlp:default` (Amendment 1 to
   6d). A reviewer can argue that the `xla:default` win is the mirror image: structure helps
   **where the flat model fails to optimize at 1e-3**. That would be a trainability effect,
   not an information-content effect, and Finding 1 claims information content.
4. **Self-consistency.** The Phase 7 audit's Axis 5 asks whether a paper's baseline is genuine
   or a strawman. This paper must pass its own rubric.

**What this design can and cannot explain.** TPUGraphs also trained its layout baseline at
**lr 0.001** (Table 5). It tuned hidden size, GNN depth and opcode-embedding size instead, and
its protocol differs from ours (config cap, eval chunking, possibly tie handling). So this
design tests **the LR-handicap explanation only**. If the flat arm stays low at every LR, the
gap to 0.12 is still **not explained**, and it stays a disclosed comparability caveat. It does
not become evidence of a strawman.

## 2. Research question

**Does the `xla:default` advantage of the structured arms over the flat arm survive tuning the
flat arm's learning rate?**

## 3. Frozen hypotheses & registered predictions

- **P1 (primary, directional).** Paired Δτ (**hardware@1e-3 − flat@LR\***) is **positive and
  significant** at α = 0.05, two-sided. LR\* is defined in §4.
- **P2 (secondary).** Paired Δτ (**hierarchical@1e-3 − flat@LR\***) is positive and
  significant, with its p Holm-adjusted jointly with P1's p. This is the hierarchy component
  of Finding 1's decomposition. The edge component (hardware − hierarchical) involves no flat
  arm and is unaffected by this design.
- **P3 (descriptive, not a pass/fail criterion).** Whether flat@LR\* reaches the published
  0.12. Recorded for context only, because the protocols are not identical (§1).

**Registered alternative, stated in advance so a failure cannot be spun:** if P1 fails, the
`xla:default` advantage is **not shown to survive baseline tuning**. Finding 1 must then be
scoped (see §7), and the paper must say that the flat arm's lr-1e-3 handicap may explain part
or all of it. **This is the outcome that most damages Finding 1, and it is registered as such.**

## 4. Data, arms, protocol

- **Collection:** `data/predict-ai-model-runtime/npz_all/npz/layout/xla/default`, the
  dataset's own `valid/` split (7 graphs), as in Phase 6.
- **New runs:** flat (`wide-baseline`) only, at **lr ∈ {1e-4, 3e-4, 3e-3}** × 5 seeds
  `{42,100,123,456,789}` = **15 runs**. Together with the existing lr-1e-3 runs, this is a
  half-decade grid spanning 1e-4 to 3e-3. Both directions are covered, because a nearly flat
  loss is consistent with an LR that is either too high or too low.
- **Protocol:** identical to Phase 6 (`launch_capped.sh`): 50 epochs, `--configs-per-batch 8`,
  `--max-configs-per-graph 128`, `--eval-batch 8`, `--dropout 0.0`, `--weight-decay 1e-4`,
  `--grad-clip 0.1`, `--eval-every` unchanged. Cosine decay lr → lr/100 (scales automatically
  with the LR). **The only change is `--lr`.**
- **Launch:** `EXTRA_ARGS="--lr <LR>" RUN_TAG=wide-baseline-lr<LR> CKPT_BASE_DIR=checkpoints/flatlr
  FOREGROUND=1 bash launch_capped.sh xla/default wide-baseline <SEED>`. The launcher's
  hardcoded `--lr 1e-3` comes before `EXTRA_ARGS`, and argparse keeps the last value. **The
  first run's log line `Fresh LR schedule: <LR> → …` must be checked before the rest proceed.**
- **Existing runs, reused unchanged:** flat@1e-3, hierarchical@1e-3 and hardware@1e-3 from
  `checkpoints/expansion/xla_default/`.
- **Selection:** `best_model.pt` → `val_tau` per run, exactly as for every other cell.
- **LR\* rule (fixed now):** LR\* is the single LR in {1e-4, 3e-4, 1e-3, 3e-3} with the
  **highest mean flat val τ across the 5 seeds**. One LR is used for all seeds, with no per-seed
  choice. An exact tie goes to 1e-3.
- **⚠️ Registered bias, deliberately in the flat arm's favour.** LR\* is chosen on the same
  validation set that scores the contrast (a winner's curse), and only the flat arm is tuned.
  Both choices **inflate flat@LR\*** and so make P1 *harder* to pass. That is intended: the
  objection is "the baseline is handicapped," so the check gives the baseline every advantage.
  The consequence, registered now: if P1 **passes**, it holds even more strongly under a fair
  comparison. If P1 **fails**, that does *not* show structure is useless, because the
  structured arms were never tuned either.
- **Boundary rule (fixed now):** if LR\* is at an edge of the grid (1e-4 or 3e-3), the optimum
  may lie beyond it. **Exactly one** pre-registered extension is then allowed: one more
  half-decade step past that edge (3e-5 or 1e-2), 5 seeds, same rules. LR\* is re-selected over
  the enlarged grid. No other LR may be added.
- **Collapse rule:** a flat run whose final `evaluate()` returns `val_n_graphs = 0`
  (constant output) is entered as **τ = 0** and flagged. No seed is replaced.

## 5. Primary metric & test

- **Metric:** Kendall's τ on `valid/`, **paired per seed**: `Δ_s = τ_structured(s) − τ_flat@LR*(s)`.
- **Primary test (P1):** paired two-sided t-test, α = 0.05, n = 5.
- **P2:** same test, Holm-adjusted jointly with P1's p. P1's pass/fail uses its **unadjusted**
  p, because it is the single pre-specified primary.
- **Reported alongside (not gates):** the full flat LR curve (mean ± sd at every LR), Wilcoxon
  signed-rank (it cannot reach 0.05 at n=5, so it indicates direction only), bootstrap CI on
  mean Δτ, Cohen's d, and the per-seed table.
- **Eval batch:** 8, as registered for all Phase 6 cells. Sensitivity to the eval batch is
  O1's job and is reported there, not here.

## 6. POWER — computed BEFORE the decision rules

`power_analysis.py mde --sigma 0.0133 0.03 0.0583 --n-seeds 5` (paired t, α=0.05, power=0.8):

| σ (paired) | Where it comes from | MDE (Δτ) |
|---|---|---|
| 0.0133 | the existing hardware − flat@1e-3 contrast | 0.022 |
| 0.030 | intermediate | 0.050 |
| **0.0583** | worst paired σ observed in this project (`nlp:random`) | **0.097** |

The σ of the new contrast is unknown, because a flat arm at another LR may be noisier, so the
worst case is used for the satisfiability check.

**Satisfiability check — PASSES, with a registered boundary.** P1 can be confirmed if the data
support it. With hardware@1e-3 at 0.198, P1 is detectable at worst-case σ as long as tuning
lifts the flat arm to no more than **≈ 0.10**. At σ ≈ 0.03 the same holds up to **≈ 0.15**.
No ceiling in this design caps the effect below the MDE (compare the Phase 8d lesson).

**⚠️ The asymmetry, registered now.** If tuning lifts the flat arm to about the published 0.12,
the remaining Δτ (≈ 0.08) falls **below** the worst-case MDE. A non-significant P1 then
**cannot** be read as "the advantage is gone." It can only be read as "not shown at n=5." In
that case the write-up **must quote the realized MDE** next to the result.

## 7. Decision rules (fixed before data)

| Outcome | Verdict | What goes in the paper |
|---|---|---|
| P1 passes **and** LR\* = 1e-3 | **Robust.** The flat arm was already at its best LR in the grid. | Finding 1 stands as registered. O5's LR component → 🟢. Report the LR curve in the appendix. |
| P1 passes, LR\* ≠ 1e-3 | **Robust, smaller.** | Finding 1 stands. **The headline Δτ becomes the tuned contrast** (hardware − flat@LR\*), with the lr-1e-3 contrast reported beside it as the originally registered number. Committed now so the smaller number cannot be dropped. |
| Δτ > 0, P1 p ≥ 0.05 | **Not shown robust to baseline tuning.** | Finding 1's `xla:default` claim is scoped to "at a shared lr of 1e-3." Quote the MDE. **No "trending toward significance."** |
| Δτ ≤ 0 | **P1 FAILS.** | The `xla:default` advantage is attributable to the flat arm's LR handicap. Finding 1 is **restated as a trainability result** (consistent with Amendment 1's `nlp:default` reading) and is not reported as information content. Registered in advance so it cannot be framed as a wash. |

P2 is reported with the same verdict wording. If P1 passes but P2 fails, the hierarchy
component of the decomposition is scoped the same way, and the edge component is unaffected.

**No outcome allows dropping this check from the paper.** The flat LR curve is reported
whichever way it falls.

## 8. Anti-p-hacking commitments

1. **n = 5 seeds, fixed.** No seed is added after seeing results. No seed is dropped as an
   outlier without reporting the analysis both with and without it.
2. **The LR grid is fixed** (§4). The only allowed extension is the single boundary step in §4.
3. **No metric substitution.** Top-K, MSE and slowdown are not fallbacks if τ disappoints.
4. **No arm substitution or re-tuning of the structured arms after the fact.** Tuning the
   structured arms would be a *new* registered experiment, not a rescue of this one.
5. **Best-checkpoint rule unchanged** (`best_model.pt` / `val_tau`).
6. **Amendments** are appended below with a date, and are added **before** the relevant data is
   read, per the 6d §8.6 convention.

## 9. Gate A checklist (`DEFENSIBILITY.md`)

- [x] Prediction registered before any data exists (§0).
- [x] Power checked and satisfiability confirmed (§6).
- [x] Arm matching: parameters, seeds and protocol are matched. Tuning effort is deliberately
      *unmatched in the flat arm's favour*, and the consequence is registered (§4).
- [x] Implementation properties: eval-mode BatchNorm uses batch statistics (O1), so the eval
      batch is held at 8 as registered, and O1 reports sensitivity separately.
- [ ] Specificity/placebo control: not applicable. This is a robustness check on a baseline,
      not a mechanism claim.

## 10. Budget

~130 s/epoch × 50 epochs ≈ **~2 h per run** (from the seed-42 log), so **15 runs ≈ 30 GPU-hours**
plus evals, and +5 runs (~10 h) if the boundary step fires. Evals may run slower while the
shared machine is under CPU load. This design competes for the GPU with the (undecided) amended
`nlp:default` sweep. Order: O1 finishes → this design → the amended sweep, if the first author approves it.
