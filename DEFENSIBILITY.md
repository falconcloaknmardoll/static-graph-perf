# Defensibility Log

**Purpose.** Checks, at every stage of the research, whether each claim survives its strongest
objection. This is a different question from [`VERIFICATION.md`](VERIFICATION.md):

| Ledger | Question it answers |
|---|---|
| `VERIFICATION.md` (C-IDs) | *Has a human independently checked this claim?* |
| `DEFENSIBILITY.md` (O-IDs) | *Would this claim survive the strongest objection a hostile reviewer could raise?* |

A claim can be human-verified and still indefensible (true, but overstated), or defensible and
not yet human-verified. Both ledgers must be green before a claim goes in the paper.

**The goal is not "cannot be rejected".** Review is noisy. The goal is that **no claim in the
paper is wrong or overstated.** Every objection must end up either *answered by evidence* or
*scoped/disclosed in the paper*.

---

## 1. Stage gates — run these at EVERY stage

### Gate A — before launching an experiment
- [ ] Is the prediction registered (or explicitly labelled exploratory) **before** data exists?
- [ ] **Power:** can n seeds detect the effect the claim needs? (`power_analysis.py mde`)
- [ ] Are the arms matched on **parameters, seeds, protocol, tuning effort** (e.g. LR)?
- [ ] Could an **implementation property** produce the effect (batch coupling, leakage,
      eval-mode behaviour, sampler order)?
- [ ] Is there a **specificity/placebo control** that would distinguish the mechanism from capacity?

### Gate B — when results land
- [ ] Did every run **actually train**? (τ exactly 0.0000 or a flat loss = check
      `val_n_graphs`/constancy before reading it as a result; see L-2026-09-24)
- [ ] Is the effect robust to **arbitrary evaluation choices** (eval batch, checkpoint rule,
      tie handling)?
- [ ] Is it driven by **one graph/seed**? Report with/without the outlier.
- [ ] Was anything **observed before a registration** that claims it was unobserved?
- [ ] Multiplicity: are corrections **actually applied** in code, not just defined?

### Gate C — before a claim is written into the paper
- [ ] Every number regenerates from a **committed script + saved checkpoint**, with a file path.
- [ ] The wording is **scoped to what was tested** (collection, hardware, task, n).
- [ ] Each objection in §2 that touches this claim is **answered or disclosed in the text**.
- [ ] The claim is also 🟢/🟡 in `VERIFICATION.md`.

### Gate D — before submission
- [ ] Claim audit complete (§3): every number in the paper traced and reproduced.
- [ ] Numbers reproduce from a **clean clone** (the anonymous mirror), not the working tree.
- [ ] An adversarial human read (someone not involved) has been done, and its points are logged here.
- [ ] Every 🔴 below is resolved or explicitly disclosed.

---

## 2. Objection register

Status: 🔴 open · 🟡 mitigated by scoping/disclosure (must appear in the paper) · 🟢 answered by evidence

| ID | Objection | Claims hit | Status | Answer / evidence / required text |
|---|---|---|---|---|
| **O1** | Eval-mode BatchNorm uses **batch statistics** (`track_running_stats=False`, all models), so a config's score depends on its eval batch-mates and **τ depends on `--eval-batch`** (hardware ckpt: 0.299 @ b=8 vs 0.335 @ b=32). Is Δτ an artifact of b=8? | Result I (xla:default Δτ), Phase 6 cells | 🔴 open, check running | `scripts/eval_batch_sensitivity.py` → `experiments/eval_batch_sensitivity_xla_default.{md,json}`. Registered comparisons are internally matched (both arms b=8). Must show Δτ holds at other b, or scope it. |
| **O2** | **LR parity:** one shared LR (1e-3) is *matched*, not *equally tuned*. The hardware arm is fragile at 1e-3 (it collapses on nlp:default), so is nlp:random's negative result (hw worse, p=.0504) an under-tuned arm? | Result I (nlp:random) | 🟡 disclose | State as a limitation; soften the nlp:random reading. No per-arm tuning was done in any cell. |
| **O3** | **Small val set:** xla:default has 7 val graphs, and inception_v3 drives ~half the per-graph Δ (0.158 → 0.090 without it). | Result I | 🟡 disclose | `experiments/phase6b_confound_checks.md`. Report with and without it, prominently. |
| **O4** | **Generality:** one dataset (TPUGraphs), one hardware target (TPU), one task (layout). | all | 🟡 scope | Scope every claim; state generality as untested. |
| **O5** | **Absolute τ is low** (hw ≈0.19 on xla:default) vs published TPUGraphs numbers. | framing | 🔴 open | Verify comparison numbers; frame the paper as controlled Δ, not SOTA. See `VERIFICATION.md` C1. |
| **O6** | **Phase 7 audit:** n=20, 75% AI-prescreened; audited authors may be reviewers. | Finding 4 | 🟡 partial | Raise human-verified share >25%; guardrails 5–6 (aggregate-only, identical rubric). |
| **O7** | **Prereg breaches/amendments** (6d §0 breached; 8d P5 unsatisfiable). | 6d, 8d | 🟡 disclose | Amendment 1 (`79f81ec`), `phase8d_power_addendum.md`. Make them visible in the paper, not an appendix footnote. |
| **O8** | **Protocol mismatch inside Result I:** Phase 2c/3 (7-val-graph protocol) vs Phase 6 sweep. | Result I | 🔴 open, author's call | Disclose or harmonise. |
| **O9** | **τ is not what practitioners care about.** | Result I | 🔴 open | Add Top-1 slowdown on xla:default (exploratory, existing checkpoints). |
| **O10** | **nlp:default cell missing** from the 2×2. | Result I | 🟡 disclose | Registered training failure (Amendment 1); scope distribution-dependence to within-xla. |
| **O11** | **Hierarchy "wins" with no mechanism.** | Result I | 🔴 open | Inspect DiffPool assignments on xla:default (`VERIFICATION.md` C7). |
| **O12** | **AI assistance** in code/analysis/audit. | all | 🟡 disclose | Disclose per MLSys policy; the author must be able to defend every design choice unaided. |
| **O13** | **8c capacity artifact?** | Finding 3 | 🟢 answered | Scramble control: channel < scramble, Δ−0.244, t-p=.006 (`phase8c_control_results.md`). |
| **O14** | **"Just use more seeds, obviously."** | Finding 1/4 | 🟢 answered | Power analysis: n=5 MDE 0.067–0.116; Δτ 0.05 needs 18 seeds (`power_analysis_results.md`). |

---

## 3. Claim audit

*Gate D item. Every number in `experiments/paper_outline.md` is traced to its source and
regenerated where possible.* **Status: NOT STARTED**. It begins once O1's check lands.

| # | Outline location | Claim / number | Source file | Reproduced? | Verdict |
|---|---|---|---|---|---|

---

## 4. Log (newest first)

Each entry records a check, what it found, and what changed as a result.

- **2026-09-26 — eval-batch sensitivity (O1).** While verifying `eval_fast.py`, τ for one
  hardware checkpoint moved 0.299 → 0.335 between eval batch 8 and 32. Cause: batch-statistic
  BatchNorm in eval mode, a deliberate project-wide choice (running stats broke on feature
  scales up to 26M). Opened O1; sensitivity re-scoring of all 15 xla:default checkpoints
  launched.
- **2026-09-26 — fast evaluator verified.** `eval_fast.py` inputs are bitwise identical to
  `train.evaluate()`. Predictions differ ≤ the original's own GPU run-to-run noise (~1e-5, from
  nondeterministic scatter-add). `scripts/verify_eval_fast.py` PASS. 4–5× faster.
- **2026-09-26 — constancy check.** Both nlp:default sweep hardware checkpoints output **one
  value for every input** (seed 42 = the head's bias), 20/20 NaN τ. Confirms Amendment 1 §10.2's
  "undefined, not zero". No erratum needed. Scan of all 75 checkpoints: no other near-zero-τ run.
- **2026-09-24/25 — nlp:default hardware collapse.** A flat loss spotted during a routine status
  check turned out to be a constant-output model. It was reproduced on GPU; aux losses were ruled
  out; the LR was identified as the cause. Sweep paused; Amendment 1 frozen **before** the
  prereg-relevant τ was read. *Lesson → Gate B item 1.*
- **2026-09-22 — attribution fix.** The outline's "xla:random / nlp p=.22/.72" mapped two
  interventions onto two collections; nlp:random is really a marginal **negative** (p=.0504).
  *Lesson → Gate C item 1.*
- **2026-09-22 — 8d P5 unsatisfiable.** The registered ceiling capped the effect below the n=5
  MDE. *Lesson → Gate A power item: check satisfiability first.*
- **2026-08-21 — BH correction never applied.** `phase6b_confound_checks.py` defined
  `bh_correct()` but never called it. *Lesson → Gate B multiplicity item.*
