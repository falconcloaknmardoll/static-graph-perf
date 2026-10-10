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
| **O1** | Eval-mode BatchNorm uses **batch statistics**, so τ depends on `--eval-batch`. Is Δτ an artifact of b=8? | Result I (xla:default Δτ), Phase 6 cells | 🟢 answered | `experiments/eval_batch_sensitivity_xla_default.{md,json}` (2026-09-27): at b=8 the re-score reproduces the registered numbers exactly (+0.1157, +0.1650). Both contrasts stay significant at every batch tested: hier − flat +0.0765 / +0.1157 / +0.1206 / +0.1260 at b = 2 / 8 / 32 / 64 (all p ≤ .003); hw − flat +0.1227 / +0.1650 / +0.1583 at b = 2 / 8 / 32 (all p < 1e-4); b=64 OOMs for the hardware arm on a 16 GB GPU. b=2 shrinks every arm (BatchNorm on 2 samples). Paper: one sentence in §4 ¶5. |
| **O2** | **LR parity:** one shared LR (1e-3) is *matched*, not *equally tuned*. The hardware arm is fragile at 1e-3 (it collapses on nlp:default), so is nlp:random's negative result (hw worse, p=.0504) an under-tuned arm? | Result I (nlp:random) | 🟡 disclose | State as a limitation; soften the nlp:random reading. No per-arm tuning was done in any cell. |
| **O3** | **Small val set:** xla:default has 7 val graphs, and inception_v3 drives ~half the per-graph Δ (0.158 → 0.090 without it). | Result I | 🟡 disclose | `experiments/phase6b_confound_checks.md`. Report with and without it, prominently. |
| **O4** | **Generality:** one dataset (TPUGraphs), one hardware target (TPU), one task (layout). | all | 🟡 scope | Scope every claim; state generality as untested. |
| **O5** | **The flat baseline is weak on xla:default** (rewritten 2026-09-27). Flat val τ = 0.033 vs TPUGraphs' published best-baseline val τ = 0.12 (Table 2); hardware (0.198) is *above* the published number. The flat loss barely moved (1.304 → 1.291 over 50 epochs). Is the +0.165 Δτ a win over a handicapped baseline, i.e. trainability rather than information content? Also a self-consistency risk against the Phase 7 audit's Axis 5 (strawman baselines). | Result I / Finding 1 (xla:default Δτ, hierarchy decomposition) | 🟢 answered (finding restated): **6e P1 FAILS** (2026-10-08); **6f: no difference detected under symmetric tuning** (Δτ +0.002, p = .88, MDE 0.037; 2026-10-09) | Phase 6e `experiments/phase6e_flat_lr_preregistration.md`: flat at lr {1e-4, 3e-4, 3e-3} × 5 seeds, primary = hardware@1e-3 − flat@LR*. It tests the **LR** explanation only; TPUGraphs also used lr 1e-3, so the rest of the gap to 0.12 remains a disclosed comparability caveat. Drop VERIFICATION C1's "beat τ=0.34" (0.34 is the xla:random *test* number; we have no test labels). **2026-10-05 provisional read (log `Best val_τ` lines; formal evaluation after the boundary step):** flat val τ by LR: 1e-4 0.2254 ± 0.027, 3e-4 0.1841 ± 0.050, 1e-3 0.0326, 3e-3 0.0261 → LR\* = 1e-4 (grid edge; boundary step lr 3e-5 running, queue h). P1 hardware@1e-3 − flat@1e-4 = −0.028, p = 0.14; P2 hierarchical − flat@1e-4 = −0.077, p = 0.004. This is §7's "Δτ ≤ 0 → P1 FAILS" row, and it holds whatever 3e-5 shows (flat@LR\* ≥ 0.225 > hardware 0.198). Finding 1 on xla:default is restated as a trainability result. All five flat@1e-4 runs peaked at epoch 50 (disclose). Follow-up: **6f** (`experiments/phase6f_hardware_lr_preregistration.md`, `9b11a96`) tunes the hardware arm on the same grid and rule, two-sided. |
| **O6** | **Phase 7 audit:** n=20, 75% AI-prescreened; audited authors may be reviewers. | Finding 4 | 🟡 partial | Raise human-verified share >25%; guardrails 5–6 (aggregate-only, identical rubric). Provenance checked 2026-09-26: the author confirmed he personally checked the 5 papers (the notes in the audit sheet are his). "Human-verified" is accurate. The session note calling it AI re-derivation was wrong and has been corrected. |
| **O15** | **How was the audit corpus chosen?** No documented search protocol or sampling frame exists. `phase7_meta_analysis.md` has a scope rule (papers making a comparative architectural claim) and a queue of candidates found by search, but no stated search terms, sources, date range, or inclusion/exclusion log. A reviewer can read this as cherry-picking. | Finding 4 | 🟡 disclose (coverage FINAL 2026-10-05: Frame A 8/30, Frame B 0/10, author-verified) | **(1) Done 2026-09-27:** retrospective selection protocol written and labelled as such (`phase7_meta_analysis.md`, "Selection protocol"). It records purposive/snowball selection, round-2 search terms not logged, and the complete exclusion list. Claims are rescoped to "a purposive sample of 20". **(2) Declared 2026-09-27, before running:** citation-ranked coverage check (Semantic Scholar, 6 fixed queries, Frame A = top-30 in-scope, Frame B = top-10 in-scope 2024–26). The overstated "every score is quote-verified" wording is **not in the repo**; it was in the advisor update, so fix it in the next message to the second author (the accurate figure is 5/20 human-verified). |
| **O16** | **The 8d pre-registration defines P5 two ways.** §3 states P5 as threshold-only (≤1.05, called "a stronger claim than 'significantly better than channel'"); §9 requires "≤1.05 **and** significantly beats channel". `phase8d_evaluate.py`'s `p5_pass: true` tests §3 only. | Finding 3 (§7) | 🟡 disclose (decided) | **Agreed 2026-09-27 (the first author):** apply §9 as the decision rule; say in §7 that the prereg states P5 two ways and that the released `p5_pass` field tests §3's threshold only. `scripts/phase8d_p5_rule_check.py` → `experiments/phase8d_p5_rule_check.json` records the §9 reading beside the frozen results (which are unchanged). **Decided 2026-09-27 (the first author): report row 3 as registered** — its registered reading (label noise as the deeper limit) is stated as the registered outcome, followed by the power analysis showing the test could not separate label noise from missing representation; 81% is descriptive only; the paper never says "the residual is representational". Background: mechanically, the only §9 row that matches is **row 3** ("Oracle ≈ channel, no significant further gain"), whose committed interpretation is "label noise / non-determinism as the deeper limit, not missing representation". The paper's current framing ("81% recovery favours representation") departs from that. Row 3 assumed a small effect; here Δ = −0.0725 is not small but the test is underpowered (power addendum). Either adopt row 3's reading, or state that row 3 matches and argue explicitly why its interpretation does not fit an underpowered, non-small effect. Do not silently use the representation framing. |
| **O17** | **Mixed training hardware inside §7's paired tests.** The 8c primary arms (baseline, channel, spatial, full) trained on GPU (`phase8c_master.log`: Device cuda); the scramble control and the 8d oracle trained on CPU during the Aug–Sep GPU outage (`phase8c_scramble_*.log`, `phase8d_master.log`: Device cpu). So P4 (channel vs scramble) and 8d's primary (oracle vs channel) pair arms trained on different hardware. Same code, seeds and data order, but CPU and GPU arithmetic differ, so the runs diverge and seed pairing is weaker than for same-hardware arms. | Finding 3 (§7) | 🟡 disclose | Found 2026-09-28 while preparing the compute statement. No evidence of bias: scramble (CPU) ≈ baseline (GPU), Δ −0.0601, p = .30. The "resume" in `phase8c_scramble_resume.log` is the sweep driver skipping finished seeds; every run trained 50 fresh epochs. Required text: the compute statement says which arms trained on CPU and why. Optional check (author's call, costs days of compute): retrain the channel arm on CPU to test P4 on matched hardware. |
| **O7** | **Prereg breaches/amendments** (6d §0 breached; 8d P5 unsatisfiable). | 6d, 8d | 🟡 disclose | Amendment 1 (`79f81ec`), `phase8d_power_addendum.md`. Make them visible in the paper, not an appendix footnote. |
| **O8** | **Protocol mismatch inside Result I:** Phase 2c/3 (xla:random) scored at `eval_batch` 32 vs 8 in Phase 6 (the only result-affecting arg difference). | Result I | 🟡 harmonised, wording changes | `experiments/eval_batch_sensitivity_xla_random.{md,json}` (2026-09-27): b=32 reproduces the originals (−0.0117 p=.22; −0.0116 p=.72). **At b=8 (the Phase 6 protocol): hier − flat −0.0235 ± 0.0214, p=.070; hw − flat −0.0241 ± 0.0665, p=.46.** Still no significant difference, but the hierarchy contrast now trends negative, so report every Result I cell at b=8 and describe xla:random as "no significant difference (p=.07), direction negative", not as a clean match. b=2 makes both significantly negative (BatchNorm artefact). Disclose: best-checkpoint selection for xla:random was done at b=32. |
| **O9** | **τ is not what practitioners care about.** | Result I | 🟢 answered | Declared before computing (`9fc0075`). `experiments/topk_slowdown_<coll>.{md,json}` (2026-09-27). **xla:default:** the gain reaches the configs an autotuner picks: Top-1 slowdown flat 1.152 ± 0.018 vs hierarchical 1.006 ± 0.002 (Δ −0.146, p = 5e-5) and hardware 1.009 ± 0.008 (Δ −0.143, p = 1e-4); significant at K = 1, 5, 10, 100. **xla:random:** null at every K (all p ≥ .45), matching Phase 4. **nlp:random** (2026-09-30, `experiments/topk_slowdown_nlp_random.{md,json}`): hardware is *worse* than flat at K = 5 (+0.0125, p = .008) and K = 10 (+0.0067, p = .025), n.s. at K = 1 (+0.019, p = .09) and K = 100; consistent with nlp:random's negative τ result. Note: flat's Top-1 error on xla:default (15.2%) is also far above TPUGraphs' published 3.8% (val), which is further O5 context. |
| **O10** | **nlp:default cell missing** from the 2×2. | Result I | 🟡 disclose | Registered training failure (Amendment 1); scope distribution-dependence to within-xla. |
| **O11** | **Hierarchy "wins" with no mechanism.** | Result I | 🟡 disclose | Disclose in the paper as a stated limitation, with the facts: (1) the xla:default config space is measurably subtler than xla:random on 7/7 graphs (`phase6c_mechanism_check.md` §1); (2) DiffPool clustering does **not** differ by collection, so the simplest story ("hierarchy clusters xla:default differently") is ruled out (§2); (3) DiffPool uses only **4 of 128** layer-1 clusters (soft, peakedness ≈0.54). (4) **Untested hypothesis:** the flat model has a strong *local* config view (dual pooling: global mean + mean over configurable nodes, ~3-hop receptive field; `model.py`), while the hierarchical models have **no** config readout but a *global* view. The win is therefore consistent with non-local layout effects on xla:default. A locality test needs training, so it is **deferred until Phase 6e's verdict**; it is pre-registered then only if 6e confirms the win and GPU time allows. |
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

- **2026-10-10 — Top-K at the tuned LRs DECLARED (committed before any number is computed).**
  Exploratory, like O9; it never replaces 6f's Q1 (τ) verdict. Why: O9's xla:default Top-K
  numbers compare against flat@1e-3, which 6e showed is LR-handicapped. Fixed now: arms =
  flat@1e-4 (`checkpoints/flatlr/…lr1e-4`) and hardware@1e-4 (`checkpoints/hwlr/…lr1e-4`), each
  arm's LR\* from 6e/6f; hierarchical is excluded (never tuned). Same metric, K ∈ {1, 5, 10, 100},
  eval batch 8, 5 paired seeds, paired two-sided t, no correction (exploratory, as O9).
  `python scripts/topk_slowdown.py --coll xla_default_tuned` → `experiments/topk_slowdown_xla_default_tuned.{md,json}`.
  Caveat to report with it: LR\* was chosen on τ over the same validation graphs. Reported
  whichever way it falls; O9's lr-1e-3 numbers are then scoped to "at a shared lr of 1e-3".
- **2026-10-09 — Phase 6f formal evaluation: §7 row 3, no difference detected under symmetric tuning (n = 5), within a 50-epoch budget.**
  `scripts/phase6f_evaluate.py`, `experiments/phase6f_results.{md,json}`. Hardware LR curve:
  3e-5 0.2024, **1e-4 0.2271 (LR\*\_hw, after the boundary step)**, 3e-4 0.1898, 1e-3 0.1977,
  3e-3 0.1493. Both arms peak at lr 1e-4. Q1 hardware@1e-4 − flat@1e-4 = +0.0016 (t = 0.16,
  p = .879, Wilcoxon p = .81, bootstrap CI [−0.017, +0.017], d_z 0.07, realized MDE 0.037).
  Q2: tuning moved hardware +0.029 (flat moved +0.193). Convergence: best epoch ≥ 46 in 4/5
  hardware and 5/5 flat runs → the "within a 50-epoch budget" qualifier applies. No collapses
  (all 20 new runs scored 7/7 graphs). O5 → 🟢: the xla:default win was the flat arm's LR
  handicap; under symmetric tuning, structure is not shown to help. Wording rule: no "trending",
  no "equivalent"; quote MDE 0.037.
- **2026-10-08 — Phase 6e formal evaluation: P1 FAILS (§7 row 4), confirmed from checkpoints.**
  `scripts/phase6e_evaluate.py`, `experiments/phase6e_results.{md,json}`. After the boundary step
  (flat 3e-5 mean 0.2067), LR\* = 1e-4 (0.2255), no longer a grid edge. P1 hardware@1e-3 −
  flat@1e-4 = −0.0278 (t = −1.84, p = .139, bootstrap CI [−0.056, −0.004], d_z −0.82, MDE 0.056);
  P2 hierarchical − flat@1e-4 = −0.0770 (p = .0037, Holm .0074). P3: tuned flat (0.2255) exceeds
  TPUGraphs' published 0.12. Matches the 2026-10-05 provisional read exactly. Finding 1 on
  xla:default → trainability result per prereg; the fair comparison waits on 6f (~Oct 8 midday).
  Wording rule: do not say hardware is "significantly worse" (t-test n.s.; only the bootstrap CI
  excludes 0).
- **2026-10-05 (late) — Axis 3 human-verified for all 20 audit papers; one correction (A2 F → ∂); headline 90% → 85%.**
  The first author checked the 15 papers not in the August sample against their PDFs (check sheet: Claude Doc
  "Axis 3 check"); Claude re-verified the factual claims in the first author's log (rendered AutoTVM Fig. 4,
  COGNATE Fig. 4; Polynormer Table 8; Omniwise Table 3). Changes:
  (1) **A2 TGraph Axis 3 F → ∂**: Table 2 reports ±1-sigma for every cell of the headline, no test
  (rubric ∂ anchor). First correction toward a looser score; all earlier ones went stricter.
  (2) **A10 AutoTVM kept F** under the rubric's "when in doubt, weaker category" rule: Figs. 4–5 show
  shaded bands that are never defined. A10's Axis 1 evidence ("single curves") was factually wrong
  and is corrected. (3) A17 evidence corrected (no StarCoder2 numbers exist; choice asserted).
  (4) Claim-accuracy notes logged for A12 and A15 (text overstates its own tables); Axis 5 scores
  NOT changed — candidate for an Axis 5 review if wanted.
  **Headline consequences:** Axis 3 = 19 F + 1 ∂ (95% F); "no paper runs a significance test" still
  20/20; papers failing ≥3 axes 18/20 (90%) → **17/20 (85%)** (A2 now fails only Axes 2, 4);
  0/20 pass Axes 1–3 unchanged. Propagated to `phase7_meta_analysis.md` §A/§D, `paper_outline.md`,
  `paper/tables/audit_axes.tex` (regenerated; ledger check passes), `08_audit.tex` markers, Desk.
  Wording rule: say "no paper runs a significance test", never "100% fail Axis 3".
- **2026-10-05 (late) — O15 coverage check FINAL: author spot-check agrees with all 31 flagged calls.**
  The first author reviewed every ambiguous/boundary decision (full abstracts, Verdict per paper) and overturned
  none. Registered result stands: **Frame A 8/30, Frame B 0/10.** §8 ¶2 can describe the screening as
  AI-assisted with every judgement call author-verified. O15 → 🟡 (disclose: purposive sample; the
  uncovered in-scope list is not audited; the frame retrieves only 10 of the 20 corpus papers).
- **2026-10-05 (late) — O15 coverage check screened (provisional until the first author's spot-check).**
  Screened the declared frame (571 candidates) per the 2026-09-27 protocol; log in
  `experiments/phase7_coverage/screening.{md,json}`. **Frame A 8/30, Frame B 0/10** (ambiguous → IN,
  as declared). Sensitivity, not registered: 10/30 and 2/10 if every ambiguous call went OUT.
  Disclosures: ~480 off-domain papers excluded on title alone (protocol says title + abstract);
  analytical-vs-learned models and model-comparing benchmarks screened IN for consistency with
  A1/A11/A14; 6 duplicate pairs merged; the frame itself retrieves only 10 of the 20 corpus papers
  (misses TPUGraphs). 31 decisions flagged for spot-check. Consequence for claims: the corpus is
  confirmed as a purposive sample, not a representative one, so the §8 wording "a purposive sample
  of 20" stands and the uncovered in-scope list is disclosed as not audited.
- **2026-10-05 (22:40) — B0 lost 2 seeds to a wandb timeout; completed in queue (j); train.py hardened.**
  Queue (h) finished seeds 42, 100, 123 at flat lr 3e-5 (best val τ 0.2154, 0.2040, 0.1894). Seed 456
  crashed inside `wandb.init` (90 s startup timeout, before epoch 1), so the LR check found no epoch-1
  line and aborted (h); seed 789 never ran. No training data was lost or read selectively: the crash
  happened before any training. The first author's decision: run seeds 456 and 789 at the head of queue (j), after 6f's
  grid (queue i), rather than pausing 6f. The formal 6e evaluation therefore waits for ~Oct 8. The P1
  verdict does not depend on B0 (flat@1e-4 already beats hardware), only its magnitude does. `train.py`
  change (logging only, training math untouched): `wandb.init` gets a 300 s timeout and falls back to
  offline mode instead of crashing, so a network blip cannot abort a queued run again. Runs launched
  from 22:40 on Oct 5 use it.
- **2026-10-05 — B1 (amended nlp:default sweep) chained after 6f; 6f boundary rule automated.**
  The first author's decision, made before any B1 data exists. Queue (j) waits for queue (i), then applies 6f's
  frozen LR\* + boundary rule via `scripts/lr_star_boundary.py` (dry-run on the finished 6e flat
  grid reproduces LR\* = 1e-4 → 3e-5; exits non-zero on any missing or unfinished log, in which
  case the step is skipped and logged for a manual check). The boundary step runs before B1 so
  the Level-1 result is not delayed. B1 runs all 13 runs as registered (6d §10.5, Amendment 2);
  ETA ~Oct 11–12, inside the Oct 23 cutoff.
- **2026-10-05 — Phase 6e read: the flat baseline was LR-handicapped; 6f pre-registered.**
  Queue (g) finished 2026-09-30 (all 15 6e runs + O9 Top-K nlp:random) but was not read until
  today. Provisional 6e read from the log `Best val_τ` lines: flat's best LR is **1e-4 (τ 0.2254
  ± 0.027)**, above hardware@1e-3 (0.198) and TPUGraphs' published 0.12. P1 (hardware − flat@LR\*)
  = −0.028, p = 0.14 → **"Δτ ≤ 0 → P1 FAILS"**; P2 (hierarchical − flat@LR\*) = −0.077, p = 0.004.
  The verdict cannot change with the boundary step (flat@LR\* ≥ 0.225 by construction), which
  only sets the magnitude. Changing flat's LR alone moved its τ 7× (0.033 → 0.225), more than the
  claimed effect (+0.165): the paper's own baseline failed Axis 5 at a shared LR, and the
  pre-registered check caught it before submission. Actions: (1) boundary step (BACKBURNER B0)
  launched as queue (h), flat lr 3e-5 × 5 seeds, LR confirmed at epoch 1; (2) **Phase 6f
  pre-registered and frozen** (`9b11a96`, pushed before any 3e-5 result was read): hardware arm on
  6e's grid and rule, two-sided, with registered fixes for unequal winner's-curse inflation, a
  stated most-likely outcome ("no difference detected"), and a 50-epoch convergence check;
  cutoff Oct 15 23:59 ET; queue (i) (`59ea92f`) starts automatically after (h). The first author chose to
  tune the structured arm (option a) over disclosure only. (3) O11's locality test (B3) loses its
  trigger ("only if 6e confirms the win"). (4) the second author: **not notified separately** (the first author's decision,
  2026-10-05): he has agreed to the direction of every step so far; this log is the record and
  goes into the next status update. *Paper consequence:* Level 1's xla:default claim is under
  revision until 6f lands (~Oct 7–8); the 6b decomposition is scoped to "at a shared lr of 1e-3".
  Note on order: a status check printed seed 42's epoch-5 val τ for the 3e-5 run *after* the
  6f freeze; nothing in 6f depended on it.

- **2026-09-27 — the registered LR check named a log line that fresh runs never print.** The
  6e pre-registration (§4) and 6d Amendment 2 (§11.3) say each run's LR is confirmed from its
  `Fresh LR schedule: …` log line, but `train.py` prints that line only when resuming. Fresh
  runs never do, so the queue would have discarded the first finished 6e run and stopped. Caught
  before any run finished. Replaced by `scripts/check_run_lr.py`, which checks the epoch-1 `lr=`
  value against train.py's cosine schedule (validated on the running lr-1e-4 log and an old
  lr-1e-3 log; rejects a wrong LR). Operational only: it verifies the same fact, and no
  protocol or decision rule changed. *Lesson → Gate A: dry-run every automated check against a
  real log before relying on it.*

- **2026-09-27 — paper numbers regenerated from source files (`scripts/paper_figures.py`).**
  Three findings. (1) **Audit per-paper tables were stale.** Recomputing from the 20 §A score
  tables gave 17/20 papers failing ≥3 axes (headline says 18/20), Axis 4 at 15% and Axis 5 at
  20% (ledger 20% / 30%). Cause: the three human-pass corrections recorded in §D (TenSet
  Axis 5, HLS-GNN Axis 5, RLM-for-Code Axis 4, all ∂→F) were never written back into §A. They
  are now, marked "∂ → F (human verification, §D)", and the recomputation matches every
  headline number. No paper was re-scored. (2) **O16 opened:** the 8d prereg defines P5 two
  ways, and the released JSON's `p5_pass: true` tests only one of them. (3) **MDE rounding:**
  the tool gives 0.0665 at σ = 0.040, n = 5, which rounds to 0.066; the docs quote "0.067–0.116".
  The paper's generated table says 0.066, so the text must too.
  *Lesson → Gate C: regenerate every paper number from the evidence trail itself, not from a
  summary of it. Summaries drift from their sources.*

- **2026-09-27 — O11: a proposed mechanism refuted by reading the code, before any run.**
  Claude proposed a "dilution" mechanism: the flat model mean-pools over all nodes, so the
  config signal carried by the few configurable nodes is diluted, and hierarchy concentrates it.
  Before declaring the check, reading `model.py` showed the flat model uses **dual pooling**
  (global mean + a dedicated mean over configurable nodes). The flat model's config signal is
  therefore *not* diluted; it is the hierarchical models that lack a config readout. The
  hypothesis was dropped without running anything. O11 → 🟡 disclose (see register). Also
  corrected VERIFICATION C7: clusters stay soft but only 4 of 128 are used.
  *Lesson → Gate A: read the architecture before proposing a mechanism for it. A mechanism
  must be checked against what the model can actually compute.*

- **2026-09-27 — O9 Top-K analysis DECLARED (committed before any number is computed).**
  Exploratory; it never replaces τ as Finding 1's test (6d §8.2 "no metric substitution").
  Fixed now: **metric** slowdown@K = best true runtime in the model's top-K / true best
  (Phase 4's definition; `scripts/topk_slowdown.py` matches `phase4_evaluate_topk.py` on 500/500
  random cases); **K ∈ {1, 5, 10, 100}** (1/5/10 for continuity with Phase 4, 100 for TPUGraphs
  Table 2); per graph, then mean over val graphs, dropping graphs with < K configs for that K;
  **eval batch 8**; existing `best_model.pt` checkpoints, not re-selected; **cells** xla:default
  (3 arms), xla:random (3 arms, 2c/3 checkpoints re-scored at b=8), nlp:random (2 arms).
  nlp:default is excluded (registered training failure). **Test:** paired two-sided t per arm
  vs flat per K, n=5, no multiplicity correction because it is exploratory and every cell is
  reported. **Commitment:** every cell and every K is reported whichever way it falls. If the
  xla:default τ win does not show at top-K, the paper says so as a scoping sentence on
  Finding 1's practical relevance.
- **2026-09-27 — O8 traced to the eval batch.** A diff of every saved arg in the 2c/3 vs
  Phase 6 checkpoints found exactly one result-affecting difference: `eval_batch` 32 vs 8. O8
  is therefore a special case of O1. The xla:random re-score is queued behind O1.
  *Lesson → Gate C: diff the saved run args, not the written protocol. The docs described
  both protocols identically.*

- **2026-09-27 — O5 re-examined against the published numbers.** Pulled TPUGraphs Table 2
  (val τ: xla:default 0.12, xla:random 0.19, nlp:default 0.30) instead of relying on memory.
  The objection as logged ("absolute τ is low") was aimed at the wrong arm: hardware (0.198)
  beats the published xla:default number, but the **flat arm (0.033) is far below it**. That
  makes the xla:default win open to a trainability reading, which is the mirror image of the
  nlp:default hardware collapse under the same shared LR. O5 rewritten. The flat-LR check is
  pre-registered as Phase 6e (draft, frozen on commit). TPUGraphs Table 5 shows they also
  trained at lr 1e-3, so an LR check cannot fully explain the gap to 0.12.
  *Lesson → Gate C: check a comparison against the source table before logging the objection;
  the shape of the objection changes what check answers it.*

- **2026-09-26 — audit sheet vs repo cross-check.** Read all 20 rows of the audit sheet and
  compared them with `phase7_meta_analysis.md`. They match everywhere except two cells where
  the sheet contradicts its own notes. **A16 (RLM-for-Code) criterion 4:** the note overrides ∂
  → F, but the check column says "Confirmed" and no human score is entered (the repo uses F,
  which matches the note). **A18 (HLS-GNN) criterion 4:** the human score says F, but the note
  says "P stands" (the repo uses P). The headline numbers are unaffected either way. Criterion
  4's fail rate is 20% if A18 is P and 25% if it is F. **Resolved the same day:** the author
  re-checked the paper himself and scored A18 criterion 4 as **P**. The split sentence is
  verbatim in §5.1, hyperparameters were tuned on validation (§5.1), there are no program
  variants (§3.2: independently generated by ldrgen), and the headline Table 5 is on real-world
  benchmarks that §5.1 says were "only used for generalization evaluation". The repo stats
  stand (criterion 4 at 20%). Only the sheet cells need fixing.
  *Lesson → Gate B: when results live in two places, diff them before quoting either.*
- **2026-09-26 — Phase 7 corpus selection (O15).** While checking what the advisor update
  says about the audit, found that no search protocol or sampling frame is documented for how
  the 20 papers were chosen. Also, the update's "every score is quote-verified" overstates it
  (15/20 are AI-extracted only). Opened O15. *Lesson → Gate A: an audit or corpus study
  needs its sampling frame written down before scoring starts.*

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
