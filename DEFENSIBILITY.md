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
| **O5** | **The flat baseline is weak on xla:default** (rewritten 2026-09-27). Flat val τ = 0.033 vs TPUGraphs' published best-baseline val τ = 0.12 (Table 2); hardware (0.198) is *above* the published number. The flat loss barely moved (1.304 → 1.291 over 50 epochs). Is the +0.165 Δτ a win over a handicapped baseline, i.e. trainability rather than information content? Also a self-consistency risk against the Phase 7 audit's Axis 5 (strawman baselines). | Result I / Finding 1 (xla:default Δτ, hierarchy decomposition) | 🔴 open, check pre-registered | Phase 6e `experiments/phase6e_flat_lr_preregistration.md`: flat at lr {1e-4, 3e-4, 3e-3} × 5 seeds, primary = hardware@1e-3 − flat@LR*. It tests the **LR** explanation only; TPUGraphs also used lr 1e-3, so the rest of the gap to 0.12 remains a disclosed comparability caveat. Drop VERIFICATION C1's "beat τ=0.34" (0.34 is the xla:random *test* number; we have no test labels). |
| **O6** | **Phase 7 audit:** n=20, 75% AI-prescreened; audited authors may be reviewers. | Finding 4 | 🟡 partial | Raise human-verified share >25%; guardrails 5–6 (aggregate-only, identical rubric). Provenance checked 2026-09-26: the author confirmed he personally checked the 5 papers (the notes in the audit sheet are his). "Human-verified" is accurate. The session note calling it AI re-derivation was wrong and has been corrected. |
| **O15** | **How was the audit corpus chosen?** No documented search protocol or sampling frame exists. `phase7_meta_analysis.md` has a scope rule (papers making a comparative architectural claim) and a queue of candidates found by search, but no stated search terms, sources, date range, or inclusion/exclusion log. A reviewer can read this as cherry-picking. | Finding 4 | 🔴 open → 🟡 once coverage lands | **(1) Done 2026-09-27:** retrospective selection protocol written and labelled as such (`phase7_meta_analysis.md`, "Selection protocol"). It records purposive/snowball selection, round-2 search terms not logged, and the complete exclusion list. Claims are rescoped to "a purposive sample of 20". **(2) Declared 2026-09-27, before running:** citation-ranked coverage check (Semantic Scholar, 6 fixed queries, Frame A = top-30 in-scope, Frame B = top-10 in-scope 2024–26). The overstated "every score is quote-verified" wording is **not in the repo**; it was in the advisor update, so fix it in the next message to the second author (the accurate figure is 5/20 human-verified). |
| **O16** | **The 8d pre-registration defines P5 two ways.** §3 states P5 as threshold-only (≤1.05, called "a stronger claim than 'significantly better than channel'"); §9 requires "≤1.05 **and** significantly beats channel". `phase8d_evaluate.py`'s `p5_pass: true` tests §3 only. | Finding 3 (§7) | 🟡 disclose (decided) | **Agreed 2026-09-27 (the first author):** apply §9 as the decision rule; say in §7 that the prereg states P5 two ways and that the released `p5_pass` field tests §3's threshold only. `scripts/phase8d_p5_rule_check.py` → `experiments/phase8d_p5_rule_check.json` records the §9 reading beside the frozen results (which are unchanged). **Decided 2026-09-27 (the first author): report row 3 as registered** — its registered reading (label noise as the deeper limit) is stated as the registered outcome, followed by the power analysis showing the test could not separate label noise from missing representation; 81% is descriptive only; the paper never says "the residual is representational". Background: mechanically, the only §9 row that matches is **row 3** ("Oracle ≈ channel, no significant further gain"), whose committed interpretation is "label noise / non-determinism as the deeper limit, not missing representation". The paper's current framing ("81% recovery favours representation") departs from that. Row 3 assumed a small effect; here Δ = −0.0725 is not small but the test is underpowered (power addendum). Either adopt row 3's reading, or state that row 3 matches and argue explicitly why its interpretation does not fit an underpowered, non-small effect. Do not silently use the representation framing. |
| **O17** | **Mixed training hardware inside §7's paired tests.** The 8c primary arms (baseline, channel, spatial, full) trained on GPU (`phase8c_master.log`: Device cuda); the scramble control and the 8d oracle trained on CPU during the Aug–Sep GPU outage (`phase8c_scramble_*.log`, `phase8d_master.log`: Device cpu). So P4 (channel vs scramble) and 8d's primary (oracle vs channel) pair arms trained on different hardware. Same code, seeds and data order, but CPU and GPU arithmetic differ, so the runs diverge and seed pairing is weaker than for same-hardware arms. | Finding 3 (§7) | 🟡 disclose | Found 2026-09-28 while preparing the compute statement. No evidence of bias: scramble (CPU) ≈ baseline (GPU), Δ −0.0601, p = .30. The "resume" in `phase8c_scramble_resume.log` is the sweep driver skipping finished seeds; every run trained 50 fresh epochs. Required text: the compute statement says which arms trained on CPU and why. Optional check (author's call, costs days of compute): retrain the channel arm on CPU to test P4 on matched hardware. |
| **O7** | **Prereg breaches/amendments** (6d §0 breached; 8d P5 unsatisfiable). | 6d, 8d | 🟡 disclose | Amendment 1 (`79f81ec`), `phase8d_power_addendum.md`. Make them visible in the paper, not an appendix footnote. |
| **O8** | **Protocol mismatch inside Result I:** Phase 2c/3 (xla:random) scored at `eval_batch` 32 vs 8 in Phase 6 (the only result-affecting arg difference). | Result I | 🟡 harmonised, wording changes | `experiments/eval_batch_sensitivity_xla_random.{md,json}` (2026-09-27): b=32 reproduces the originals (−0.0117 p=.22; −0.0116 p=.72). **At b=8 (the Phase 6 protocol): hier − flat −0.0235 ± 0.0214, p=.070; hw − flat −0.0241 ± 0.0665, p=.46.** Still no significant difference, but the hierarchy contrast now trends negative, so report every Result I cell at b=8 and describe xla:random as "no significant difference (p=.07), direction negative", not as a clean match. b=2 makes both significantly negative (BatchNorm artefact). Disclose: best-checkpoint selection for xla:random was done at b=32. |
| **O9** | **τ is not what practitioners care about.** | Result I | 🟢 answered | Declared before computing (`9fc0075`). `experiments/topk_slowdown_<coll>.{md,json}` (2026-09-27). **xla:default:** the gain reaches the configs an autotuner picks: Top-1 slowdown flat 1.152 ± 0.018 vs hierarchical 1.006 ± 0.002 (Δ −0.146, p = 5e-5) and hardware 1.009 ± 0.008 (Δ −0.143, p = 1e-4); significant at K = 1, 5, 10, 100. **xla:random:** null at every K (all p ≥ .45), matching Phase 4. nlp:random pending (queued last). Note: flat's Top-1 error on xla:default (15.2%) is also far above TPUGraphs' published 3.8% (val), which is further O5 context. |
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
