# Phase 8e — Pre-registration: is the 8c channel effect robust to LR tuning?

> **Status: FROZEN 2026-10-11 (the first author sign-off, this date). No 8e run existed at freeze.** Drafted by Claude from
> `experiments/framing_audit_2026-10-11.md` (Task 1, R1). Freeze = the first author's sign-off (the second author review
> recommended), recorded at the bottom, then commit, then launch. Any later change is a dated
> amendment, added before the relevant data is read (6d §8.6 convention).

## 0. Observation status (the anti-HARKing artifact)

- **No baseline or channel run on the 8c pipeline exists at any LR other than 1e-3.**
  `checkpoints/phase8c/xla_random/` holds only `mxu-{baseline,channel,spatial,full,scramble}-seed*`,
  all at lr 1e-3 (`launch_capped.sh:122`; checkpoint args). No `checkpoints/phase8e/` exists.
- **Already known, used as motivation only:** every 8c / 8c-control / 8d result at lr 1e-3; the
  8c training logs; the collapse diagnostic (`experiments/phase8c_collapse_diagnostic.json`);
  6e/6f (flat and hardware LR curves on xla:default).

## 1. Motivation (with evidence)

- 8c (pre-registered) found channel narrows CONV Top-1 slowdown from 1.4246 to 1.1203
  (Δ −0.3043, BH-p .0032, d −3.15; `phase8c_results.md:6–7,12`), confirmed against a scramble
  control (P4, Δ −0.2442, p = .0063; `phase8c_control_results.md:29`).
- 6e showed that the same flat model (`wide-baseline`) under the same registered lr 1e-3 recipe was
  handicapped on xla:default (0.0327 at 1e-3 vs 0.2255 at LR\* = 1e-4; `phase6e_results.md:18,20`).
  That made an apparent 5/5-seed, p < .001 structural win dissolve (6e P1 fails; 6f no difference).
- The 8c baseline is that model at that LR. It trained on xla:random, and it does **not** show 6e's
  signature (loss falls 1.27–1.36 → 1.08–1.12; val τ 0.21–0.27). But (audit §1c):
  1. 4/5 baseline and 5/5 channel runs peak at epoch 50 (under-converged within budget);
  2. MXU features are config-invariant, so any effect acts through training dynamics
     (`phase8c_preregistration.md:119–122`);
  3. the baseline ranks CONV configs in the wrong direction (mean Spearman(pred, runtime) +0.048 on
     xla:default and +0.120 on xla:random, where good ranking is ≪ 0);
  4. channel also raises **overall** xla:random val τ (mean 0.3326 vs 0.2311), which looks like an
     optimization change, not a CONV-local one.
- So whether part of the 8c effect is a baseline-LR artifact is undetermined (audit verdict:
  "can't tell"). The paper's own §5 argument would be turned on §7. This design decides it.

## 2. Research question

Does `channel` still beat `baseline` on CONV Top-1 slowdown when **both** arms are run at their own
best LR from the same grid, under the same rule?

## 3. Hypotheses & registered predictions

- **Q1 (primary, two-sided).** Paired Δ = CONV Top-1 (**channel@LR\*_ch − baseline@LR\*_b**). Is
  it ≠ 0 at α = 0.05? We **predict Δ < 0 and significant** (the effect survives), but the design is
  two-sided and all outcomes are pre-committed (§7).
- **Q2 (secondary, 6e-style asymmetric).** Paired Δ = CONV Top-1 (**channel@1e-3 − baseline@LR\*_b**):
  does the registered 8c contrast survive tuning only the baseline? Holm-adjusted jointly with Q1;
  Q1's verdict uses its unadjusted p because it is the single pre-specified primary.
- **Registered expectation, stated so it can't be spun later:** the most likely bad outcome is
  that tuning lifts the baseline on CONV and the gap shrinks below the MDE ("not shown robust").

## 4. Data, arms, protocol (fixed now)

- **Arms:** `baseline` (158-dim) and `channel` (162-dim), exactly as 8c (`train_8c.py`,
  `dataset_mxu.py`, `--mxu-arm`). Spatial, full and scramble are **not** re-run. P4 stays an
  lr-1e-3 result and is reported as such.
- **Training:** xla:random train, 50 epochs, cosine LR → LR/100, wd 1e-4, grad-clip 0.1,
  dropout 0, configs-per-batch 8, max-configs-per-graph 128, eval batch 8, num_workers 0, capped
  harness (`launch_capped.sh`), GPU. LR passed as `EXTRA_ARGS="--lr <lr> --mxu-arm <arm>"` (the 6f
  mechanism; the last `--lr` wins). It is verified from the epoch-1 line by `scripts/check_run_lr.py`
  before the next run starts.
- **Seeds:** {42, 100, 123, 456, 789}, paired by seed. Arms interleaved within each seed (as 8c).
- **LR grid:** {1e-4, 3e-4, 1e-3}. The **1e-3 cells reuse the existing 8c checkpoints**
  (`checkpoints/phase8c/xla_random/mxu-{baseline,channel}-seed*`). Disclosed: they were trained in
  July 2026 with the same code and harness. If the software stack changed since then, list the
  versions in the appendix.
- **Selection (unchanged from 8c):** `best_model.pt` by overall val τ on `xla/random/valid`.
- **LR\* rule (fixed now):** for each arm separately, LR\*_arm is the single LR with the **highest
  mean overall val τ on xla/random/valid across the 5 seeds** (the stored `val_tau`). It is **never**
  chosen on CONV Top-1, the outcome. One LR per arm, no per-seed choice. An exact tie goes to 1e-3.
- **Boundary rule (fixed now):** if LR\*_arm is at a grid edge (1e-4 or 1e-3), exactly **one**
  half-decade step past that edge is run for that arm (3e-5 or 3e-3), 5 seeds, same rules, and
  LR\*_arm is re-selected over the enlarged grid. No other LR may be added.
- **Collapse rule:** a run whose final evaluation returns `val_n_graphs = 0` is entered as τ = 0
  for LR\* selection and flagged. Its CONV Top-1 is computed as-is. No seed is replaced.
- **⚠️ Registered bias:** LR\* is selected on xla/random/valid, which contains the xla:random copies
  of the CONV graphs (3 of 7 graphs). Selection uses overall τ, not CONV Top-1, so the bias is
  indirect. Both arms get the same advantage. Disclosed in the write-up.

## 5. Primary metric & test

- **Metric:** CONV-subset Top-1 slowdown on the frozen 8c split (`phase8c_preregistration.md:45–48`),
  both xla valid sets, 6 instances, averaged per seed. Same evaluator as 8c (`phase8c_evaluate.py`),
  extended to read 8e checkpoints. **No metric substitution.**
- **Test:** paired two-sided t-test, n = 5, α = 0.05. Reported alongside (not gates): Wilcoxon
  (floor .0625 at n = 5, so direction only), bootstrap 95% CI, Cohen's d, realized MDE, per-seed table.
- **Report either way (registered now, not gates):**
  1. **overall xla:random val τ for every arm × LR** (mean ± sd, per seed). It tests whether channel's
     +0.10 overall-τ jump persists at LR\*;
  2. OTHER-subset Top-1 placebo, channel@LR\*_ch − baseline@LR\*_b;
  3. per-instance CONV table (6 instances × both arms at LR\*);
  4. convergence: best epoch per run. If ≥ 3/5 runs of an arm at LR\* peak at epoch ≥ 46, the
     verdict carries "within a 50-epoch budget" (the 6f rule);
  5. the collapse-diagnostic CONV Spearman(pred, runtime) for both arms at LR\*.
- **FIG:** CONV Top-1 and overall τ vs LR, both arms, seeds as dots. Reported whichever way it falls.

## 6. Power (computed before the decision rules)

`power_analysis.py mde --sigma 0.0966 0.1132 --n-seeds 5` (paired t, α = 0.05, power = 0.8):

| σ (paired, CONV Top-1) | Source | MDE at n = 5 |
|---|---|---|
| 0.0966 | 8c channel − baseline | 0.1606 |
| 0.1132 | worst paired σ in the 8c family (scramble − baseline) | 0.1882 |

**Satisfiability: PASSES with a registered boundary.** Slowdown is ≥ 1.0 by definition, so the
largest possible |Q1| is baseline@LR\*_b − 1.0. Q1 is detectable only if the tuned baseline's CONV
slowdown stays above **≈ 1.16–1.19**. The current 1.4246 leaves room. If tuning brings the
baseline below ≈ 1.19, **no outcome** can reach significance at n = 5, whatever channel does.
In that case the verdict is "not shown robust", read with the MDE, and **not** "the effect is
gone" (this mirrors 6e §6's asymmetry and avoids 8d's unsatisfiable-threshold trap).

## 7. Decision rules (fixed before data)

| Outcome (Q1) | Verdict | What goes in the paper |
|---|---|---|
| Δ < 0, p < .05 | **8c effect survives symmetric LR tuning.** | §7 headline becomes the tuned contrast (channel@LR\*_ch − baseline@LR\*_b), with the registered lr-1e-3 contrast beside it. The audit's "can't tell" → answered. The conv result may lead. |
| Δ < 0, p ≥ .05 | **Not shown robust to LR tuning.** | §7 claims are scoped to "at a shared lr of 1e-3". Quote the MDE. No "trending". The conv result does not lead. |
| Δ ≥ 0 | **The 8c advantage is attributable to LR / training dynamics.** | §7 is restated the way Finding 1 was. 8c is reported as not robust to tuning, and P4 is scoped to lr 1e-3. Registered now so it can't be framed as a wash. |

Q2 is reported with the same verdict wording. If Q1 passes and the overall-τ jump disappears at
LR\*, report both. That outcome supports a CONV-specific effect.

**No outcome allows dropping 8e from the paper.** Both LR curves are reported.

## 8. Anti-p-hacking commitments

1. n = 5 seeds, fixed. None added, none dropped without reporting both analyses.
2. The LR grid is fixed. The only extension is the single boundary step per arm.
3. No metric or arm substitution. Top-5/Top-10 and spatial/full are not fallbacks.
4. Best-checkpoint rule unchanged.
5. **Cutoff: every registered run (grid + any triggered boundary step) must finish by Oct 23 2026,
   23:59 ET. Otherwise 8e is reported as not completed and Q1/Q2 are not evaluated.** No partial
   evaluation, so the outcome can't decide when to stop.
6. Amendments are dated and appended before the relevant data is read.

## 9. Gate A checklist (`DEFENSIBILITY.md`)

- [x] Registered before any 8e data exists (§0).
- [x] Power checked, satisfiability stated with its boundary (§6).
- [x] Arm matching: same backbone, seeds, recipe and grid. Tuning effort is symmetric.
- [x] Implementation: eval batch held at 8 (O1). LR verified from the epoch-1 line.
- [x] Specificity: the overall-τ and OTHER-placebo checks are registered as report-either-way (§5).

## 10. Budget

8c GPU runs took 9,764–12,866 s each (`logs/capped/phase8c_master.log`), so ≈ 3 h/run on an idle
GPU. New runs: 2 arms × 2 LRs × 5 seeds = **20 runs ≈ 60 GPU-h**. Boundary step: +5 runs per arm
whose LR\* sits on an edge, ≈ +15 h each. Worst case ≈ 90 h ≈ 3.75 days. To make the Oct 23 cutoff
with slack on a shared GPU: **freeze and launch by Oct 15.** The queue script is written after
freeze, modelled on `scripts/run_gpu_queue_2026-10-05i.sh` (LR check + pause switch).

## Freeze

- [x] the first author sign-off (date): 2026-10-11 (given in session; recorded by Claude)
- [ ] the second author reviewed (optional): not before freeze; the first author chose to freeze and launch to protect the Oct 23 cutoff
