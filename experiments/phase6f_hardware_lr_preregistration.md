# Phase 6f — Pre-registration: does structure help on xla:default when BOTH arms are LR-tuned?

> **Registered 2026-10-05, BEFORE any hardware run at a non-1e-3 LR existed.** Drafted by Claude
> at the first author's instruction; the first author is the registering author and approved the design. **Frozen on
> commit and pushed right away** so the timestamp is on the server. **The commit must land before the queue (h) results (flat lr
> 3e-5, Phase 6e boundary step) are read**, and no run in this design may start before it.
> Conventions follow `phase6e_flat_lr_preregistration.md`, `phase6d_nlp_default_preregistration.md`
> and `phase8c_preregistration.md`. Standing rule (`project-prereg-rigor-framing`): a
> pre-registered result is never softened after the fact, and **"was it powered?" is asked
> before "how do I phrase this?"** (§6 comes before §7). Bears on objections **O5** and **O2** in
> `DEFENSIBILITY.md`.
>
> **This is a NEW experiment, not a rescue of 6e.** 6e §8 item 4 states that tuning the
> structured arms "would be a *new* registered experiment". 6e's verdict is reported exactly as
> registered, whatever this design finds.

## 0. Observation status — the anti-HARKing artifact

At drafting time (verified 2026-10-05):
- **No `hardware` run on `xla:default` exists at any LR other than 1e-3.** The directory
  `checkpoints/hwlr/` does not exist, and no `xla_default_hardware*lr*` log exists. The outcome
  measure for every new arm in this design is therefore **unobserved**.
- **Already known, and used below as motivation (not as outcome data):**
  - hardware@1e-3 on `xla:default`, 5 seeds (Phase 6): **0.1977 ± 0.0133**. Best epochs 45, 20,
    50, 45, 50, so 3 of 5 runs peaked at or near the last epoch.
  - **Phase 6e flat runs at lr {1e-4, 3e-4, 3e-3}, provisional read of 2026-10-05** (log
    `Best val_τ` lines, not yet the formal evaluation): flat 1e-4 **0.2254 ± 0.0273**, 3e-4
    0.1841 ± 0.0497, 1e-3 0.0326 ± 0.0136, 3e-3 0.0261 ± 0.0027. LR\*\_flat = 1e-4, a grid edge,
    so 6e's boundary step fired.
  - Provisional 6e P1: hardware@1e-3 − flat@1e-4 = −0.028, p = 0.14 (paired sd ≈ 0.034).
- **Running at drafting time, NOT read:** queue (h), flat@3e-5 × 5 seeds (6e boundary step,
  launched 2026-10-05 13:31). Its result decides LR\*\_flat under 6e's rule. This design takes
  LR\*\_flat **as 6e defines it**, whatever it turns out to be.
- **Related but not data for this design:** `nlp:default` hardware collapsed at lr 1e-3 (6d
  Amendment 1). The amended lr-1e-4 `nlp:default` sweep (BACKBURNER B1) has not been run.

## 1. Motivation

6e tuned only the flat arm, deliberately in its favour (6e §4). Its registered consequence: if P1
fails, "that does *not* show structure is useless, because the structured arms were never tuned
either." P1 failed in the provisional read (Δτ ≤ 0). So the question "does structure help on
`xla:default`?" is **open in both directions**. Finding 1 can currently say only that a tuned
flat baseline matches or beats *untuned* structure, and a reviewer will ask the obvious next
question. Three facts make it a fair question rather than a fishing trip:
1. The flat arm's τ moved 7× across one half-decade grid, so this benchmark is highly
   LR-sensitive. Nothing says lr 1e-3 is best for the hardware arm either.
2. The hardware arm's own best epochs (45–50 for 3 of 5 seeds) leave room for a different LR to
   help.
3. LR was also the lever in the `nlp:default` hardware collapse.

## 2. Research question

**Under symmetric LR tuning (same grid, same selection rule, same budget), does the hardware
DiffPool arm rank `xla:default` layouts better than the flat arm?**

## 3. Frozen hypotheses

No directional prediction is registered. 6e moved our prior, and the honest prior is now
uncertain in both directions. The test is two-sided, and every outcome has a fixed verdict (§7).

- **Q1 (primary).** Paired Δτ = **hardware@LR\*\_hw − flat@LR\*\_flat**, two-sided paired t,
  α = 0.05, n = 5.
- **Q2 (secondary, descriptive, no test).** hardware@LR\*\_hw − hardware@1e-3: how much tuning
  moved the structured arm.

## 4. Data, arms, protocol

- **Collection:** `xla/default`, the dataset's own `valid/` split (7 graphs), as in 6 and 6e.
- **New runs:** `hardware` only, at **lr ∈ {1e-4, 3e-4, 3e-3}** × seeds `{42,100,123,456,789}`
  = **15 runs**. With the existing hardware@1e-3 runs, this is **the same grid 6e gave the
  flat arm**. Equal tuning budget is the point of the design, so the grid is not trimmed.
- **Protocol:** identical to 6e (`launch_capped.sh`): 50 epochs, `--configs-per-batch 8`,
  `--max-configs-per-graph 128`, `--eval-batch 8`, `--dropout 0.0`, `--weight-decay 1e-4`,
  `--grad-clip 0.1`, cosine decay lr → lr/100. DiffPool auxiliary losses unchanged. **The only
  change is `--lr`.** Device: CUDA (RTX 5080), the same as hardware@1e-3 and all 6e flat runs,
  so O17's mixed-hardware issue does not arise.
- **Launch:** `EXTRA_ARGS="--lr <LR>" RUN_TAG=hardware-lr<LR> CKPT_BASE_DIR=checkpoints/hwlr
  FOREGROUND=1 bash launch_capped.sh xla/default hardware <SEED>`. Each run's LR is confirmed
  from its epoch-1 `lr=` line (`scripts/check_run_lr.py`) before the queue continues.
- **Existing runs, reused unchanged:** hardware@1e-3 (`checkpoints/expansion/xla_default/`),
  and all flat runs of 6e (`checkpoints/flatlr/` plus flat@1e-3 in `checkpoints/expansion/`).
- **Selection:** `best_model.pt` → `val_tau` per run, exactly as for every other cell.
- **LR\*\_hw rule (fixed now; mirrors 6e):** the single LR in {1e-4, 3e-4, 1e-3, 3e-3} with the
  **highest mean hardware val τ across the 5 seeds**. One LR for all seeds. An exact tie goes to
  1e-3.
- **Boundary rule (mirrors 6e):** if LR\*\_hw is 1e-4 or 3e-3, **exactly one** extension step
  (3e-5 or 1e-2), 5 seeds, same rules, and LR\*\_hw is re-selected over the enlarged grid. No
  other LR may be added.
- **LR\*\_flat:** taken from 6e unchanged (its grid, its boundary step, its rule).
- **Collapse rule (mirrors 6e):** a run whose final `evaluate()` returns `val_n_graphs = 0` is
  entered as **τ = 0** and flagged. No seed is replaced, and a collapsed LR still enters the
  LR\* means.
- **⚠️ Registered bias: applied to both arms, but NOT equal in size.** Both LR\*s are chosen on
  the same validation set that scores the contrast (winner's curse; there are no test labels).
  Both arms get the same grid, rule and boundary step, but the inflation from a max over LR
  means grows with an arm's seed variance and with how close its top LRs are. The flat arm's
  per-LR sd is larger (0.050 at 3e-4, 0.027 at 1e-4) than the hardware arm's at 1e-3 (0.013), so
  the selection may inflate flat@LR\*\_flat more. The direction and size are unknown in advance.
  The bias is disclosed, not corrected.
- **Out of scope, registered now:** the plain `hierarchical` arm is **not** tuned. The 6b
  decomposition (hierarchy vs edge machinery) therefore stays scoped to "at a shared lr of 1e-3",
  whatever Q1 shows. The epoch budget (50) is not tuned either; see the convergence check below.
- **Convergence check (registered now; it scopes the verdict, it does not gate it).** At a fixed
  budget of 50 epochs, a low LR may leave an arm unconverged, so the contrast can partly measure
  which arm converges faster. For each arm at its LR\*, report the number of seeds whose best
  epoch is **≥ 46** (the last 5 epochs). If that holds for **≥ 3 of 5 seeds in either arm**, the
  §7 verdict is stated as **"within a 50-epoch budget"**, and the convergence counts are
  reported next to it. The 6e evaluation reports the same counts for flat@LR\*\_flat.

## 5. Primary metric & test

- **Metric:** Kendall's τ on `valid/`, paired per seed:
  `Δ_s = τ_hw@LR*_hw(s) − τ_flat@LR*_flat(s)`.
- **Q1 test:** paired two-sided t-test, α = 0.05, n = 5. Single primary, so there is no
  multiplicity adjustment.
- **Reported alongside (not gates):** the hardware LR curve next to the flat LR curve (mean ± sd
  at every LR, one figure), Wilcoxon signed-rank (direction only at n=5), bootstrap CI on mean
  Δτ, Cohen's d, the per-seed table, and the best epoch per run.

## 6. POWER — computed BEFORE the decision rules

`power_analysis.py mde --sigma 0.0133 0.0273 0.03 0.0583 --n-seeds 5` (paired t, α=0.05,
power=0.8):

| σ (paired) | Where it comes from | MDE (Δτ) |
|---|---|---|
| 0.0133 | hardware − flat, both @1e-3 (Phase 6) | 0.022 |
| 0.0273 | flat@1e-4 between-seed sd (6e, provisional) | 0.045 |
| ≈0.034 | hardware@1e-3 − flat@1e-4 paired sd (6e, provisional) | ≈0.056 |
| **0.0583** | worst paired σ observed in this project (`nlp:random`) | **0.097** |

**Satisfiability check — PASSES, in both directions.** With flat@LR\*\_flat ≥ 0.225 (6e
guarantees this, because LR\*\_flat maximises the flat mean), Q1 can show "structure helps" if
tuned hardware reaches about **0.28** at σ ≈ 0.034 (about 0.32 at worst case). It can show "flat
is better" if tuned hardware stays about 0.056 below flat (already nearly the provisional
untuned gap). τ is bounded by 1, so no ceiling caps the effect below the MDE (compare the 8d
lesson).

**Most likely outcome, stated now.** "Satisfiable" means detectable *in principle*. A "structure
helps" verdict needs tuned hardware about +0.08 above its lr-1e-3 value. Flat's 7× jump came
from an LR that was badly wrong for it, whereas hardware@1e-3 already trained reasonably (0.198,
sd 0.013), so a jump that size is not expected. **The most likely outcome is the third row of
§7, "no difference detected".** That row is a substantive answer to this design's question: it
removes the "structure was never tuned" caveat from Finding 1. It is not a failed experiment.

**⚠️ Registered asymmetry.** A difference smaller than the realized MDE (≈0.05–0.10) **cannot
be detected at n = 5**. A non-significant Q1 is "no difference detected", never "equivalent".
The write-up **must quote the realized MDE** next to any non-significant result.

## 7. Decision rules (fixed before data)

| Outcome | Verdict | What goes in the paper |
|---|---|---|
| Δτ > 0, p < 0.05 | **Structure helps under symmetric tuning.** | Level 1's information-content claim is restored **in tuned form**: the headline Δτ is this contrast. 6e's P1 failure is still reported as registered, as the evidence that the shared-LR comparison was handicapped. The 6b decomposition stays scoped to lr 1e-3. |
| Δτ < 0, p < 0.05 | **A tuned flat GNN beats tuned structure.** | Level 1 states that, on `xla:default`, the structured arms cost ranking quality once both are tuned. No "structure is competitive" wording. |
| p ≥ 0.05, either sign | **No difference detected under symmetric tuning (n = 5).** | Level 1 states that structure is not shown to help once both arms are tuned, quoting the MDE. **No "trending", no equivalence claim.** |

Q2 is reported descriptively under every outcome. If the §4 convergence check triggers, every
verdict above is qualified with "within a 50-epoch budget".

**No outcome allows dropping this check, or 6e, from the paper.** Both LR curves are reported
whichever way they fall.

## 8. Anti-p-hacking commitments

1. **n = 5 seeds, fixed.** No seed is added after seeing results. No seed is dropped as an
   outlier without reporting the analysis both with and without it.
2. **The LR grid is fixed** (§4). The only allowed extension is the single boundary step.
3. **No metric substitution.** Top-K, MSE and slowdown are not fallbacks if τ disappoints.
4. **No further arms or knobs.** Tuning `hierarchical`, the epoch budget, weight decay or width
   would each be a new registered experiment, not a rescue of this one.
5. **Best-checkpoint rule unchanged** (`best_model.pt` / `val_tau`).
6. **Cutoff:** all runs, including any boundary step, must be `.done` by **2026-10-15 23:59 ET**
   so the result reaches the Oct 16 full draft. If they are not, the design is reported as
   **registered but not completed**, and partial results are not reported as a finding.
7. **Amendments** are appended below with a date, before the relevant data is read (6d §8.6
   convention).

## 9. Gate A checklist (`DEFENSIBILITY.md`)

- [x] Design registered before any outcome data exists (§0). The flat boundary step is running
      but unread, and LR\*\_flat is fixed by 6e's rule, not chosen here.
- [x] Power checked and satisfiability confirmed in both directions (§6).
- [x] Arm matching: seeds, protocol, device, LR grid, selection rule and boundary rule are
      matched. Parameters are as in Phase 6 (unchanged). The untuned `hierarchical` arm is
      registered as out of scope (§4).
- [x] Implementation properties: eval-mode BatchNorm uses batch statistics (O1), so the eval
      batch is held at 8, as registered for every Phase 6 cell.
- [ ] Specificity/placebo control: not applicable. This is a symmetric-tuning robustness check,
      not a mechanism claim.

## 10. Budget

~180 s/epoch × 50 epochs ≈ **~2.5 h per run** (hardware@1e-3 log, quiet GPU), so **15 runs ≈
38 GPU-hours** plus evals, and +5 runs (~12.5 h) if the boundary step fires. Evals may run
slower on the shared machine. Order: queue (h) (6e boundary step) → this design. If launched by
Oct 6, it should finish around Oct 8–9 (Oct 10 with the boundary step), before BACKBURNER B1's
~Oct 18 latest start. The same pause switch (`PAUSE_GPU_QUEUE`) applies.
