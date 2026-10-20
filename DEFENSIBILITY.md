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
- [~] ~~An adversarial human read (someone not involved) has been done, and its points are logged here.~~ **Dropped 2026-10-19 (the first author): no time before the deadline.** Peer review serves as the outside read; the claim audit and the clean-clone check still stand.
- [ ] Every 🔴 below is resolved or explicitly disclosed.

---

## 2. Objection register

Status: 🔴 open · 🟡 mitigated by scoping/disclosure (must appear in the paper) · 🟢 answered by evidence

| ID | Objection | Claims hit | Status | Answer / evidence / required text |
|---|---|---|---|---|
| **O1** | Eval-mode BatchNorm uses **batch statistics**, so τ depends on `--eval-batch`. Is Δτ an artifact of b=8? | Result I (xla:default Δτ), Phase 6 cells | 🟢 answered | `experiments/eval_batch_sensitivity_xla_default.{md,json}` (2026-09-27): at b=8 the re-score reproduces the registered numbers exactly (+0.1157, +0.1650). Both contrasts stay significant at every batch tested: hier − flat +0.0765 / +0.1157 / +0.1206 / +0.1260 at b = 2 / 8 / 32 / 64 (all p ≤ .003); hw − flat +0.1227 / +0.1650 / +0.1583 at b = 2 / 8 / 32 (all p < 1e-4); b=64 OOMs for the hardware arm on a 16 GB GPU. b=2 shrinks every arm (BatchNorm on 2 samples). Paper: one sentence in §4 ¶5. |
| **O2** | **LR parity:** one shared LR (1e-3) is *matched*, not *equally tuned*. The hardware arm is fragile at 1e-3 (it collapses on nlp:default), so is nlp:random's negative result (hw worse, p=.0504) an under-tuned arm? | Result I (nlp:random) | 🟡 disclose | State as a limitation; soften the nlp:random reading. [Updated 2026-10-11: per-arm LR tuning now exists on xla:default (6e flat, 6f hardware; both LR* = 1e-4) and is running for 8c's arms (8e). xla:random and nlp:random remain single shared LR; nlp:default amended runs are LR-matched at 1e-4, not tuned.] |
| **O3** | **Small val set:** xla:default has 7 val graphs, and inception_v3 drives ~half the per-graph Δ (0.158 → 0.090 without it). | Result I | 🟡 disclose | `experiments/phase6b_confound_checks.md`. Report with and without it. [Updated 2026-10-11: under option A the inception_v3 analysis is an lr-1e-3 effect and lives in the appendix; §4 ¶6 points there.] |
| **O4** | **Generality:** one dataset (TPUGraphs), one hardware target (TPU), one task (layout). | all | 🟡 scope | Scope every claim; state generality as untested. |
| **O5** | **The flat baseline is weak on xla:default** (rewritten 2026-09-27). Flat val τ = 0.033 vs TPUGraphs' published best-baseline val τ = 0.12 (Table 2); hardware (0.198) is *above* the published number. The flat loss barely moved (1.304 → 1.291 over 50 epochs). Is the +0.165 Δτ a win over a handicapped baseline, i.e. trainability rather than information content? Also a self-consistency risk against the Phase 7 audit's Axis 5 (strawman baselines). | Result I / Finding 1 (xla:default Δτ, hierarchy decomposition) | 🟢 answered (finding restated): **6e P1 FAILS** (2026-10-08); **6f: no difference detected under symmetric tuning** (Δτ +0.002, p = .88, MDE 0.037; 2026-10-09) | Phase 6e `experiments/phase6e_flat_lr_preregistration.md`: flat at lr {1e-4, 3e-4, 3e-3} × 5 seeds, primary = hardware@1e-3 − flat@LR*. It tests the **LR** explanation only; TPUGraphs also used lr 1e-3, so the rest of the gap to 0.12 remains a disclosed comparability caveat. Drop VERIFICATION C1's "beat τ=0.34" (0.34 is the xla:random *test* number; we have no test labels). **2026-10-05 provisional read (log `Best val_τ` lines; formal evaluation after the boundary step):** flat val τ by LR: 1e-4 0.2254 ± 0.027, 3e-4 0.1841 ± 0.050, 1e-3 0.0326, 3e-3 0.0261 → LR\* = 1e-4 (grid edge; boundary step lr 3e-5 running, queue h). P1 hardware@1e-3 − flat@1e-4 = −0.028, p = 0.14; P2 hierarchical − flat@1e-4 = −0.077, p = 0.004. This is §7's "Δτ ≤ 0 → P1 FAILS" row, and it holds whatever 3e-5 shows (flat@LR\* ≥ 0.225 > hardware 0.198). Finding 1 on xla:default is restated as a trainability result. All five flat@1e-4 runs peaked at epoch 50 (disclose). Follow-up: **6f** (`experiments/phase6f_hardware_lr_preregistration.md`, `9b11a96`) tunes the hardware arm on the same grid and rule, two-sided. |
| **O6** | **Phase 7 audit:** n=20, 75% AI-prescreened; audited authors may be reviewers. | Finding 4 | 🟡 partial | Raise human-verified share >25%; guardrails 5–6 (aggregate-only, identical rubric). Provenance checked 2026-09-26: the author confirmed he personally checked the 5 papers (the notes in the audit sheet are his). "Human-verified" is accurate. The session note calling it AI re-derivation was wrong and has been corrected. |
| **O15** | **How was the audit corpus chosen?** No documented search protocol or sampling frame exists. `phase7_meta_analysis.md` has a scope rule (papers making a comparative architectural claim) and a queue of candidates found by search, but no stated search terms, sources, date range, or inclusion/exclusion log. A reviewer can read this as cherry-picking. | Finding 4 | 🟡 disclose (coverage FINAL 2026-10-05: Frame A 8/30, Frame B 0/10, author-verified) | **(1) Done 2026-09-27:** retrospective selection protocol written and labelled as such (`phase7_meta_analysis.md`, "Selection protocol"). It records purposive/snowball selection, round-2 search terms not logged, and the complete exclusion list. Claims are rescoped to "a purposive sample of 20". **(2) Declared 2026-09-27, before running:** citation-ranked coverage check (Semantic Scholar, 6 fixed queries, Frame A = top-30 in-scope, Frame B = top-10 in-scope 2024–26). **(3) 2026-10-12: Option B chosen.** Axis-3-only check of the 10 Frame B papers, reported separately; declaration **FROZEN e12c51a** (`experiments/axis3_recent_coverage_declaration.md`; see log). Scoring Oct 19–22, results by Oct 23. The overstated "every score is quote-verified" wording is **not in the repo**; it was in the advisor update, so fix it in the next message to the second author (the accurate figure is 5/20 human-verified). |
| **O16** | **The 8d pre-registration defines P5 two ways.** §3 states P5 as threshold-only (≤1.05, called "a stronger claim than 'significantly better than channel'"); §9 requires "≤1.05 **and** significantly beats channel". `phase8d_evaluate.py`'s `p5_pass: true` tests §3 only. | Finding 3 (§7) | 🟡 disclose (decided) | **Agreed 2026-09-27 (the first author):** apply §9 as the decision rule; say in §7 that the prereg states P5 two ways and that the released `p5_pass` field tests §3's threshold only. `scripts/phase8d_p5_rule_check.py` → `experiments/phase8d_p5_rule_check.json` records the §9 reading beside the frozen results (which are unchanged). **Decided 2026-09-27 (the first author): report row 3 as registered** — its registered reading (label noise as the deeper limit) is stated as the registered outcome, followed by the power analysis showing the test could not separate label noise from missing representation; 81% is descriptive only; the paper never says "the residual is representational". Background: mechanically, the only §9 row that matches is **row 3** ("Oracle ≈ channel, no significant further gain"), whose committed interpretation is "label noise / non-determinism as the deeper limit, not missing representation". The paper's current framing ("81% recovery favours representation") departs from that. Row 3 assumed a small effect; here Δ = −0.0725 is not small but the test is underpowered (power addendum). Either adopt row 3's reading, or state that row 3 matches and argue explicitly why its interpretation does not fit an underpowered, non-small effect. Do not silently use the representation framing. |
| **O17** | **Mixed training hardware inside §7's paired tests.** The 8c primary arms (baseline, channel, spatial, full) trained on GPU (`phase8c_master.log`: Device cuda); the scramble control and the 8d oracle trained on CPU during the Aug–Sep GPU outage (`phase8c_scramble_*.log`, `phase8d_master.log`: Device cpu). So P4 (channel vs scramble) and 8d's primary (oracle vs channel) pair arms trained on different hardware. Same code, seeds and data order, but CPU and GPU arithmetic differ, so the runs diverge and seed pairing is weaker than for same-hardware arms. | Finding 3 (§7) | ✅ answered for P4 (R3); disclose CPU arms | Found 2026-09-28 while preparing the compute statement. **Not tested at matched hardware** (reworded 2026-10-11: the earlier "no evidence of bias" leaned on scramble ≈ baseline, Δ −0.0601, p = .30, a null whose MDE is 0.188 at n = 5 and would need ~30 seeds to detect that difference; the scramble arm's OTHER-placebo wobble, +0.0079 p = .028, is the one placebo pairing CPU vs GPU, so O17 is a candidate cause). The "resume" in `phase8c_scramble_resume.log` is the sweep driver skipping finished seeds; every run trained 50 fresh epochs. Required text: the compute statement says which arms trained on CPU and why. ~~Optional check: retrain the channel arm on CPU~~ **Queued 2026-10-11 (the first author): R3**, the scramble arm re-run on GPU (`r3_scramble_gpu_replication_declaration.md`, queue m after 8e and R4, 15 GPU-h, lr 1e-3, cutoff Oct 23). It also measures the CPU-vs-GPU offset directly. The 8d oracle is not re-run (descriptive only). **2026-10-15:** O19 shows the scramble OTHER wobble is absent at eval batch 8 (−0.0008, p .89), so it is at least partly an eval-batch effect. R3's 5 runs trained (done 2026-10-15 02:08). **2026-10-16 — R3 evaluated (`r3_results.md`, row 1): P4 replicates at matched hardware** (channel − GPU scramble, CONV b32 −0.2212, p = .0033, MDE 0.131; b8 −0.2330, p = .0036). Hardware offset GPU − CPU scramble: CONV −0.023 (p = .61, MDE 0.155), OTHER −0.001 (p = .94). GPU scramble's OTHER placebo +0.0072, p = .48: the wobble does not recur. Wording now "replicated at matched hardware"; the compute statement still names the CPU-trained registered arms (8c scramble, 8d oracle). The 8d oracle is not re-run at lr 1e-3; R5 trains it on GPU at 3e-5 (`r5_oracle_tuned_declaration.md`), which bears on 8d's CPU caveat only descriptively. |
| **O18** | **Is the 8c MXU effect a baseline-LR artifact?** (framing audit 2026-10-11) The 8c baseline is the same flat `wide-baseline` at the same untuned lr 1e-3 that 6e showed handicapped on xla:default. It trains on xla:random (overall τ 0.2311, loss falls normally, so not 6e's signature), but 4/5 runs peak at epoch 50, the MXU features are config-invariant (any effect acts through training dynamics), the baseline ranks CONV configs backwards (ρ +0.048 / +0.120), and channel also lifts overall τ by ~+0.10. The paper's own §5 argument applies. | Finding 3 (§7), and the option-A contrast | 🟡 answered: largely yes (8e, 2026-10-14) | Verdict "can't tell" (`framing_audit_2026-10-11.md` §1). Test: 8e (`phase8e_lr_robustness_preregistration.md`, DRAFT, freeze + launch by Oct 15, cutoff Oct 23). Until then: §7 says the 8c baseline was not LR-tuned, option A uses the xla:random like-for-like contrast, and the conv result does not lead. **8e EVALUATED 2026-10-14** (`experiments/phase8e_results.{md,json}`): LR\* = 3e-5 for both arms (lower edge of the enlarged grid, disclose). Q1 channel@3e-5 − baseline@3e-5 CONV Top-1 −0.0331, p = .17, MDE 0.073 → registered verdict **"Not shown robust to LR tuning"**; §6 boundary breached (tuned baseline 1.0904 < ≈1.16–1.19), so read with the MDE, never "the effect is gone". Q2 channel@1e-3 − baseline@3e-5 +0.0299, p = .69, MDE 0.26 → "attributable to LR / training dynamics" (registered wording). Batch-8 sensitivity: Q1 −0.0602, p = .045, but the OTHER placebo also moves (+0.0111, p = .044); registered batch is 32. Descriptive: tuning the baseline alone takes CONV Top-1 1.4246 → 1.0904; channel ≤ baseline at all four LRs; at LR\* the remaining gap sits on resnet50.4x4.fp16 (xla:random) 1.43 vs 1.19. §7 per prereg row 2: claims scoped to a shared lr of 1e-3, MDE quoted, conv result does not lead; P4 and 8d scoped to lr 1e-3. |
| **O19** | **8c/8d Top-1 eval batch is undisclosed and its sensitivity unchecked.** `phase8c_evaluate.py` / `phase8d_evaluate.py` score CONV Top-1 at batch **32** (default; no flag passed), while all Phase 6 τ cells use batch 8. Eval-mode BatchNorm uses batch statistics (O1), and O1's check covered only xla:default τ. | Finding 3 (§7), 8d | 🟢 answered for 8c/P4 (2026-10-15); 🟡 disclose: 8d primary is eval-batch-sensitive | Found 2026-10-11 while writing R0. Required text: state batch 32 in §7 / the 8c table caption. Cheap check (eval-only, existing checkpoints, exploratory): re-score 8c baseline/channel/scramble and 8d at b = 8 (and 2/64 where they fit), as O1 did. R0 and 8e were amended before any data to score at 32 with a b = 8 sensitivity column. **Declared + queued 2026-10-13** (`experiments/o19_8c8d_batch8_declaration.md`, frozen 5c19ad5): eval-only b=8 re-score of 8c (5 arms) + 8d, registered b=32 stays primary, reproduction gate first; queue o (`scripts/run_gpu_queue_2026-10-13o.sh`) runs after R0 (~Oct 15). 8e/R3 batch-8 cells read its caches. **Evaluated 2026-10-15** (`o19_8c8d_batch8_results.md`): gate 6/210 cells differ, max |Δ| 1.2e-05. Every 8c primary row and P4 keep their verdict at b8 (channel −0.3023, BH-p .0046; P4 −0.1921, p .0077) → "8c/P4 verdicts are unchanged at eval batch 8". Placebos that change: P4's scramble OTHER wobble disappears at b8 (+0.0079 p .028 → −0.0008 p .89); 8d OTHER placebo flips sign, n.s. both. **8d primary oracle − channel becomes significant at b8** (−0.0725 p .19 → −0.0986 p .020, MDE 0.098) → the 8d paragraph must say the 8d primary is eval-batch-sensitive; registered b32 (n.s., P5 not confirmed) stays primary. |
| **O20** | **§7's CONV subset is 3 graphs from two families** (Inception; ResNet-50 ×2). Does the 8c effect generalise beyond them? | Finding 3 (§7) | 🟡 disclose (R4 done: not shown to generalise) | The validation split holds only these 3 conv-dominated graphs; test labels are not public. **Post hoc (2026-10-11):** leave-one-graph-out, the effect survives every drop (Δ −0.2597 to −0.3783, all p ≤ .0099, 5/5 seeds; `phase8c_leave_one_graph_out.md`), and it improves 6/6 graph×collection instances. The model-free sensitivity gap is on 95 conv graphs. **R4** (`r4_heldout_conv_declaration.md`, exploratory, queue l after 8e): hold out all 6 non-Inception/ResNet conv families (11 graphs, 19 instances), retrain baseline + channel at each arm's 8e LR* (Amendment 1; both 3e-5), 5 seeds; cutoff Oct 23. All 10 runs trained by 2026-10-14 16:16. **2026-10-16 — R4 evaluated (`r4_results.md`, row 2): direction consistent on held-out families, not significant at n = 5** (b32 Δ −0.1128, CI [−0.2955, +0.0699], p = .35, MDE 0.393; OTHER placebo −0.006, p = .44). **The sign flips at batch 8** (+0.1311, p = .31): disclose in the same sentence. Two instances carry the b32 effect (retinanet.4x4.bf16 −1.55, alexnet −0.89). Uncorrected LOFO at b8, drop retinanet: +0.269, p = .042 (channel worse; 1 of 12 exploratory tests, recorded here only). Post hoc: tuned baseline 1.30 on the unseen families vs 1.09 on 8c's three. The evaluation was killed mid-run 2026-10-15 16:53 (session end, no error) and resumed from its per-cell cache (`checkpoints/r3r4_eval_cache/`); only the missing cell was recomputed. Required text: name the two-family limit in §4 ¶6 and §7. |
| **O21** | **Four more analyses scored at eval batch 32, undisclosed** (research audit 2026-10-14, R1): Phase 5 per-graph Top-1 (`phase5_error_analysis.py:125`; §6 ¶1, FIG-C1), Phase 6b per-graph τ (`scripts/phase6b_confound_checks.py:39`; O3's inception_v3 share), the tightening curve (`power_analysis.py tighten` on the 6b JSON, flat arm; FIG-M2), Phase 4 xla:random Top-K (`phase4_evaluate_topk.py:93`). | §4, §5, §6 | 🟡 disclose | Rows in the eval-batch appendix table (O1/O8/O19 item), labelled batch 32. Tightening: flat arm only, per-seed means within 0.0003 of batch 8, disclosure only. Phase 4: cite `topk_slowdown_xla_random.json` (O9, batch 8, all p ≥ .45) instead. Phase 5: R0's declared batch-8 sensitivity covers the same checkpoints' per-graph Top-1; quote it once R0 is evaluated (do not read R0 caches before). 6b: see the batch-8 re-score decision (log 2026-10-14). **Done 2026-10-15:** 6b re-scored at b8 (`phase6b_confound_checks_b8.md`, gate passed): inception_v3 +0.568 → +0.550, other 6 +0.090 → +0.101 (no material change), BH unchanged → O3 quotes b8. FIG-M2 regenerated at b8 (`experiments/figures/power_tightening_curve_b8.png`; median SD 0.0145 at n = 7 vs 0.0152 at b32, so the caption gives both). Phase 5 covered: R0's b8 per-graph Top-1 (same checkpoints) is in `r0_conv_top1_results.json`. |
| **O22** | **The same flat recipe, run twice, differs by 0.057 τ.** §5's flat (Phase 2c, val τ 0.288 at b8) and §7's 8c baseline (0.231) are the same model on byte-identical inputs (`TPUGraphDataset` vs `MXUGraphDataset(baseline)`, checked 2026-10-14); only the harness differs (num_workers 4 vs 0 changes the config-sampling stream; `train_8c.py` restates the loop). Per seed −0.080/+0.036/−0.078/−0.058/−0.105, paired p = .08. | §5 vs §7 | 🟡 disclose | No verdict affected: every paired contrast is within one harness. Never quote §5's flat and §7's baseline as one number; the outline's "fair contrast" (hier −0.0235 vs channel −0.3043) must say the two contrasts come from separate runs. One sentence in §7 (baseline's own 0.231; a separate run of the same recipe gave 0.288). Fits C9: run-level randomness ≈ the n = 5 MDE (0.066). |
| **O23** | **Every 8d oracle checkpoint is from epoch 5.** All 5 seeds peaked at the first validation (val τ 0.79–0.80) and drifted to ≈ 0.77 by epoch 50 (`logs/capped/xla_random_oracle-bucket_seed*.log`). | §7 (8d) | 🟡 disclose | Same selection rule as every arm (best overall val τ), so not a protocol breach. One clause in §7 ¶5 so "5 epochs vs 50" is not read as an unequal comparison. **2026-10-19 (R5):** the GPU-trained oracle at lr 3e-5 peaks at epochs 15–40 (val τ 0.78–0.79), so epoch 5 is a property of the lr-1e-3 CPU run, not of the oracle input. |
| **O24** | **The tuned baseline is tuned, but not competitive with the state of the art.** Official-split validation τ of our best flat model per collection: xla:default 0.2255 (lr 1e-4, 6e), xla:random 0.4068 (3e-5, 8e), nlp:default 0.4138 (1e-4, 6d amended), nlp:random 0.7129 (1e-3, untuned). The published TPUGraphs baseline is 0.12 / 0.19 / 0.30 / 0.58, so ours is well above it; TGraph (Khizbullin et al., Front. AI 2025, audited in §8) reports 0.4785 / 0.6840 / 0.5628 / 0.9713, and even its "− graph edges" ablation (0.3631 / 0.5022 / 0.3349 / 0.7751) beats our xla numbers. | Whole paper (L1, L2), §4, §10 | 🟡 disclose + scope | Found 2026-10-19 (Claude, on the first author's request). Protocols differ, so not a like-for-like comparison: TGraph uses 20-fold CV on merged train+val (5 folds trained, top 4 kept), a fold ensemble plus 10× test-time augmentation, 750–1000 epochs (ours 50, untuned budget), cross-configuration attention, and its τ formula is not fully stated; ours is the official valid split, one model per seed, mean of 5 seeds, all configs per graph, per-graph τ averaged (`train.py` evaluate). Xu et al. 2023's numbers are disputed by TGraph's authors, so do not cite them as a target. **The risk:** the paper's thesis is that weak baselines manufacture gains; a reviewer will turn it around: maybe structure or MXU features help a SOTA-strength model, or the pattern only holds for a mid-strength backbone. **Required text:** §4 places the tuned baseline against both published numbers in one sentence (above the TPUGraphs baseline, below TGraph, protocols differ); §10 limitations scope L1/L2 to "a matched, LR-tuned mid-strength GNN within a 50-epoch budget", untested at SOTA strength. Note for the prose: TGraph's own ablation shows removing the base graph's edges costs 0.10–0.23 τ, so never let L1 read as "graph connectivity doesn't matter"; L1 is about *adding* hierarchy or contention edges to an already-connected graph. `phase6e_results.md` P3's "the tuned baseline is no longer weak" holds only relative to the TPUGraphs baseline. |
| **O7** | **Prereg breaches/amendments** (6d §0 breached; 8d P5 unsatisfiable). | 6d, 8d | 🟡 disclose | Amendment 1 (`79f81ec`), `phase8d_power_addendum.md`. Make them visible in the paper, not an appendix footnote. |
| **O8** | **Protocol mismatch inside Result I:** Phase 2c/3 (xla:random) scored at `eval_batch` 32 vs 8 in Phase 6 (the only result-affecting arg difference). | Result I | 🟡 harmonised, wording changes | `experiments/eval_batch_sensitivity_xla_random.{md,json}` (2026-09-27): b=32 reproduces the originals (−0.0117 p=.22; −0.0116 p=.72). **At b=8 (the Phase 6 protocol): hier − flat −0.0235 ± 0.0214, p=.070; hw − flat −0.0241 ± 0.0665, p=.46.** Still no significant difference, but the hierarchy contrast now trends negative, so report every Result I cell at b=8 and describe xla:random as "no significant difference (p=.07), direction negative", not as a clean match. b=2 makes both significantly negative (BatchNorm artefact). Disclose: best-checkpoint selection for xla:random was done at b=32. **Re-selection at b=8 checked 2026-10-13** (`experiments/o8_checkpoint_reselection_check_2026-10-13.md`): not possible without retraining (~36 GPU-h; only `best_model.pt` was kept). 12/15 selected checkpoints are the final epoch (seed 456 = epoch 45 in all 3 arms), and the b=8 penalty is arm-level (hier/hw ≈ −0.015, flat ≈ −0.003), so the selection rule barely mattered. Keep the disclosure and add the 12/15 fact. In the paper, O1/O8/O19 form one eval-batch item (methods sentence + appendix table). |
| **O9** | **τ is not what practitioners care about.** | Result I | 🟢 answered | Declared before computing (`9fc0075`). `experiments/topk_slowdown_<coll>.{md,json}` (2026-09-27). **xla:default:** the gain reaches the configs an autotuner picks: Top-1 slowdown flat 1.152 ± 0.018 vs hierarchical 1.006 ± 0.002 (Δ −0.146, p = 5e-5) and hardware 1.009 ± 0.008 (Δ −0.143, p = 1e-4); significant at K = 1, 5, 10, 100. **xla:random:** null at every K (all p ≥ .45), matching Phase 4. **nlp:random** (2026-09-30, `experiments/topk_slowdown_nlp_random.{md,json}`): hardware is *worse* than flat at K = 5 (+0.0125, p = .008) and K = 10 (+0.0067, p = .025), n.s. at K = 1 (+0.019, p = .09) and K = 100; consistent with nlp:random's negative τ result. Note: flat's Top-1 error on xla:default (15.2%) is also far above TPUGraphs' published 3.8% (val), which is further O5 context. |
| **O10** | **nlp:default cell missing** from the 2×2. | Result I | 🟡 disclose (amended cell filled 2026-10-11) | Registered result = training failure under the shared recipe (Amendment 1, §10.4), reported first. Amended §10.5 (exploratory, LR-matched at 1e-4, all 13 runs before cutoff): hardware − flat Δτ −0.0051, p = .53, MDE 0.027 → §7 Δτ < 0 row (A-P1 and A-P2 fail, as registered). Under option A this extends the null to a fourth collection; don't stretch the −0.005 sign into a model-family effect. `experiments/phase6d_amended_results.md`. |
| **O11** | **Hierarchy "wins" with no mechanism.** | Result I | 🟡 disclose | Disclose in the paper as a stated limitation, with the facts: (1) the xla:default config space is measurably subtler than xla:random on 7/7 graphs (`phase6c_mechanism_check.md` §1); (2) DiffPool clustering does **not** differ by collection, so the simplest story ("hierarchy clusters xla:default differently") is ruled out (§2); (3) DiffPool uses only **4 of 128** layer-1 clusters (soft, peakedness ≈0.54). (4) **Untested hypothesis:** the flat model has a strong *local* config view (dual pooling: global mean + mean over configurable nodes, ~3-hop receptive field; `model.py`), while the hierarchical models have **no** config readout but a *global* view. The win is therefore consistent with non-local layout effects on xla:default. A locality test needs training, so it is **deferred until Phase 6e's verdict**; it is pre-registered then only if 6e confirms the win and GPU time allows. |
| **O12** | **AI assistance** in code/analysis/audit. | all | 🟡 disclose | Disclose per MLSys policy; the author must be able to defend every design choice unaided. |
| **O13** | **8c capacity artifact?** | Finding 3 | 🟢 answered | Scramble control: channel < scramble, Δ−0.244, t-p=.006 (`phase8c_control_results.md`). |
| **O14** | **"Just use more seeds, obviously."** | Finding 1/4 | 🟢 answered | Power analysis: n=5 MDE 0.066–0.116 (0.067 corrected 2026-10-11); Δτ 0.05 needs 18 seeds (`power_analysis_results.md`). |

---

## 3. Claim audit

*Gate D item. Every number in `experiments/paper_outline.md` is traced to its source and
regenerated where possible.* **Status: NOT STARTED** (O1's check landed 2026-09-27, so it is unblocked). **Booked 2026-10-11 for Oct 17–18 (slipped); re-booked 2026-10-19 for one sitting on Oct 22–23, after R5 is applied and §7 is drafted.** Claude traces every number in the draft to its source file and line; the first author fixes the prose. Gate D needs it before submission.

| # | Outline location | Claim / number | Source file | Reproduced? | Verdict |
|---|---|---|---|---|---|

---

## 4. Log (newest first)

Each entry records a check, what it found, and what changed as a result.

- **2026-10-19 — Baseline strength vs published TPUGraphs results (O24 opened).**
  - Our tuned flat model is ~2× the TPUGraphs paper's baseline on xla (0.23 vs 0.12; 0.41 vs 0.19) but about half of TGraph on xla:default (0.48), and below TGraph's no-edges ablation on both xla collections. Protocols differ (CV + ensemble + TTA + 15× the epochs). Disclose and scope; no run can close this before the deadline.
  *Lesson → Gate A: "tuned" means tuned within our recipe; check absolute strength against the field before framing a paper around baseline strength.*
- **2026-10-19 — R5 evaluated (Gate B): row 2, "no headroom detected after tuning".**
  - `r5_results.md` (declaration 58cb565, evaluator 71ffa73): tuned oracle − tuned baseline, CONV, lr 3e-5, batch 32, n = 10: Δ −0.0432, 95% CI [−0.0812, −0.0086], t-p = .056, MDE 0.062; satisfiable (ceiling 0.085). Batch 8, same sign: −0.0926, p = .0026. OTHER placebo: oracle worse, +0.020, p < .001 at both batches (a loss, so the "not conv-specific" flag does not fire; disclose). Three of ten seeds carry the batch-32 effect.
  - Applied per declaration §6 row 2: §7 ¶5–6 scoped to "a tuned flat model is within the MDE of a model given runtime ranks"; §6's information-gap claim rests on the 2.77× sensitivity and the failed static explanations, not on model error. 8e, 8d (P5 not confirmed), P4 unchanged. O23 note added (R5 oracle peaks at epochs 15–40). Desk ¶5 FLAG answered; outline bullet added.
  - Watch in the prose: the bootstrap CI excludes zero but the deciding t-test gives p = .056. Report the CI; never use it to call the result significant or "trending".
  - The first evaluation died in a machine crash (00:46, no cache written yet) and was re-run from scratch with the same committed evaluator under a 24 G memory cap; finished 03:40. Training (15 runs) had completed Oct 17, all before the Oct 23 cutoff.
  *Lesson → Gate B: launch every evaluation under the memory cap, not only training (the crashed run had none).*
- **2026-10-16 — Gate C: full-draft read after R3/R4; oracle (§7 ¶5–6) framing; Desk refreshed.**
  - Read the whole Google Doc draft (6,777 words) against the Oct 15 framing. Seven stale spots found, all in the first author's prose (abstract findings clause; intro discoveries + contributions (i)/(iii); intro 90% → 85%; §5 heading "distribution dependent" contradicts its own ¶1/¶6; §4 ¶6 labels R4 as leave-one-graph-out; §4 ¶7 lr 1e-4 typo; audit ¶4 missing the 8c worked example). Logged as a checklist in the Desk revision log for the first author; not edited by Claude (manuscript authorship rule).
  - Desk/Primer stale lines fixed (claim lines only): §2 ¶2 "distribution-dependent" pointer; §4 O3 line (R4 ≠ leave-one-graph-out, R4 result); §8 ¶4 worked examples 2 → 3 (adds 8c −0.30 → −0.03); Oct 10 option-A item marked superseded for L2; Primer Finding 3 heading + R3 note. Desk §7: R3 and R4 slots filled; FLAGs added on ¶2 (spatial/full only at lr 1e-3) and ¶5 (oracle's 81% recovery uses the lr-1e-3 baseline).
  - **Oracle framing (Claude's recommendation, given to the first author):** do not reframe the lr-1e-3 oracle as "headroom after tuning". That compares an untuned oracle (lr 1e-3, CPU, epoch-5 checkpoints) with a tuned baseline (3e-5): cross-run and cross-LR (O22's rule), post hoc, and an untuned oracle is not a ceiling. Scope ¶5–6 to lr 1e-3, lead with P5's unsatisfiability as the audit's point on our own pre-registration, keep 81% out of the main text. The first author chose to test the question properly instead → R5 (entry below). ¶5–6 wait for R5; ¶1–4 do not depend on it.
  - **Holistic-coherence point recorded:** after 8e, §6's "information gap" rests on sensitivity (2.77×) and three failed static explanations; its model-based support was lr-1e-3 only (tuned baseline reaches 1.09). R5 is the only like-for-like test of it. Its decision table (§6) fixes in advance how §6 is scoped whichever way it falls.
  *Lesson → Gate C: when a registered result is re-scoped (8e), re-check every downstream claim that leaned on the old scope, including ones in other sections (§6's gap claim leaned on lr-1e-3 model error).*
- **2026-10-16 — R3 and R4 evaluated (Gate B); R5 declared, frozen and launched (Gate A).**
  - **R3** (`r3_results.md`): row 1, P4 replicates at matched hardware (b32 −0.2212, p = .0033; b8 −0.2330, p = .0036). O17 answered for P4; the scramble OTHER wobble does not recur on GPU (+0.0072, p = .48). Still an lr-1e-3 result.
  - **R4** (`r4_results.md`): row 2, direction consistent, n.s. (b32 −0.1128, p = .35, MDE 0.393), sign flips at b8 (+0.1311, p = .31). O20 stays disclosed. R4's evaluator was killed by the session ending; resumed from cache (results committed ec94a19).
  - **R5** (`r5_oracle_tuned_declaration.md`, frozen 58cb565; queue bbc7e99 as systemd service `gpu-queue-r5`; evaluator 71ffa73 committed before scoring): tuned oracle vs tuned baseline, lr 3e-5, n = 10. Motivated by 8e (exploratory in origin, said so in the declaration). Power checked first: at n = 5 the design was unsatisfiable (MDE 0.108 > ceiling 0.090, driven by the tuned baseline's two weak seeds), so n = 10 (MDE ≈ 0.065; 80% power needs the tuned oracle ≲ 1.025).
  *Lesson → Gate A: run the satisfiability check on every new contrast before choosing n, including follow-ups designed in a hurry. Design A at n = 5 would have repeated P5's trap.*
  *Lesson → Gate B: long evaluations must run detached from the session (systemd service), and cache per cell so a kill costs one cell.*
- **2026-10-15 — Framing decided after 8e (Gate C).**
  - The first author decided (1) unify the paper around baseline strength: both shared-LR gains (connectivity +0.165 τ; MXU feature −0.30 CONV Top-1) shrink below detection once the baseline is tuned (6e/6f; 8e). (2) §6 leads with the model-free 2.77×; Phase 5's lr-1e-3 model error comes second with the tuned baseline's 1.09 beside it.
  - Taken while R4 was still being evaluated, on purpose: R4 is exploratory and cannot lift 8e's registered §7 constraints, and choosing the framing after seeing it would invite the post-hoc reading the audit criticises.
  - Applied to claim lines only (Desk skeleton, Primer, Manuscript Desk tab, Google Doc [[SLOT]]s, outline). The first author rewrites the abstract, §1 ¶5 and §10 in his own words; the second author sees the framing before the Oct 16 draft.
  *Lesson → Gate C: when two results share a failure mode (an untuned baseline), report them as one pattern rather than one win and one caveat.*
- **2026-10-15 — R0, O19 and O21 evaluated; R3/R4 evaluation started.**
  - All GPU queues (k–p) finished overnight with no errors or OOM (R0's earlier OOM did not recur).
  - **R0** (`r0_conv_top1_results.md`, exploratory, partially observed): hierarchical − flat CONV Top-1 +0.0216 (p .77, MDE 0.257), contention-edge − flat +0.0026 (p .97, MDE 0.283) at b32; b8 the same. Declaration §4 row 1: "adding structure does not narrow the CONV gap; adding MXU information does" (lr 1e-3; quote the MDE, which is close to 8c's 0.30, so "no detectable narrowing"). Makes the option-A contrast like-for-like on metric and recipe; 8e's LR verdict still scopes the MXU half.
  - **O19**: 8c/P4 unchanged at b8; 8d primary becomes significant at b8 (p .020) → disclose as eval-batch-sensitive in the 8d paragraph. **O21**: no material change; O3 and FIG-M2 move to b8 numbers.
  - R3 and R4 had no evaluator; `scripts/r3_r4_evaluate.py` written to the frozen declarations (comparator cells read from the registered 8c and O19 caches, never recomputed; R4 asserts LR* = 3e-5 from `phase8e_lr_star.py` and 19 instances).
  *Lesson → Gate B: a queue that ends in "Next: evaluate X" needs its evaluator written before the queue finishes, or results sit unread.*
- **2026-10-14 — Gate C on Doc §5 ¶5→¶6 framing + §4 LR sentence.**
  - The first author asked whether ¶5 (xla:default lr-1e-3 'win') followed by ¶6 (tuning dissolves it) reads as an admission plus post-hoc justification. Finding: the structure is defensible because 6e was registered before the outcome and the shared LR is a disclosed design choice. It is the audit §8 ¶4 worked example. Risks to avoid: no rescue of the win in ¶6 (lr-1e-3 decomposition stays appendix-only, scoped); ¶5 should say 'passes Axes 1–3', not 'statistically convincing'; name the pre-registration at ¶6 open; keep the weak-flat flag (0.033 vs TPUGraphs 0.12; tuning moved flat +0.193 vs hw +0.029).
  - Doc §4 says 'all arms training at lr 1e-4'; the shared LR is 1e-3 (only nlp:default amended runs use 1e-4). Flagged to the first author to fix (his prose).
  *Lesson → Gate C: a result overturned by a registered check is reported as a registered failure, never paired with a rescue.*
- **2026-10-14 — 8e evaluated (O18): the 8c conv effect is not shown robust to LR tuning.**
  - Evaluated in a GPU window between R4 runs (`scripts/run_8e_eval_window.sh`, 01:38–05:02); 8c reproduction gate 137/140 cells exact, max |Δ| 2e-6.
  - Q1 −0.0331, p = .17 (MDE 0.073) → "Not shown robust"; tuned baseline 1.0904 is below the §6 satisfiability boundary, so no n = 5 outcome could have been significant. Q2 +0.0299, p = .69.
  - Post hoc, from 8e's registered per-instance and curve outputs: the baseline's conv failure (1.42) mostly disappears with LR tuning, which bears on §6 ¶1 (Phase 5's model errors came from lr-1e-3 models). Framing decisions logged in the Desk revision log, not made here.
  *Lesson → Gate A: when a design check (6e) exposes an untuned baseline, apply the same check to every result that shares that baseline before writing it up. 8e did this; the 8c framing should have stayed provisional until it landed.*

- **2026-10-14 — Research consistency audit (`experiments/paper_consistency_audit_2026-10-13.md`).**
  - Scope: data splits, evaluator code, saved args of all 161 checkpoints, every evaluation script behind a reported number, results JSONs recomputed from per-seed values. No verdict changes.
  - Opened O21 (four more undisclosed batch-32 evaluations), O22 (two runs of the same flat recipe differ by 0.057), O23 (8d oracle checkpoints all from epoch 5).
  - Consistent: no train/valid overlap within or across collections; one τ implementation; 6e/6f/6d/Top-K/8c/P4/8d reproduce from per-seed data.
  - 8e evaluated in a GPU window (queues paused between R4 runs): `scripts/phase8e_evaluate.py`, `scripts/run_8e_eval_window.sh`.
  *Lesson → Gate C: grep every evaluator for its default eval batch, not only the training args; four analyses inherited batch 32 from function defaults.*

- **2026-10-13 — O8 re-selection checked; O19 declared and queued.**
  - **O8:** the xla:random checkpoints cannot be re-selected at eval batch 8 without retraining, because each run kept only `best_model.pt`. Post-hoc, re-selection could not change either verdict: 12/15 picks are the final epoch, and the batch-8 penalty is arm-level. Disclosure kept (`experiments/o8_checkpoint_reselection_check_2026-10-13.md`).
  - **O19:** batch-8 re-score of 8c/8d declared before any data (5c19ad5), queued after R0. Found while writing it: `phase8c_evaluate.py`'s cache filename has no batch tag, so `--batch-size 8` would have silently reloaded the batch-32 results. The new script writes to batch-tagged caches.
  *Lesson → Gate C: when a result is cached, check that the cache key includes every argument that changes the result.*

- **2026-10-13 — Gate C on §5 figures: FIG-R2 showed batch-32 numbers; FIG-R1 titles contradicted option A. Both regenerated.**
  - **Found** while adding §5 figure placeholders to the Google Doc draft. FIG-R2
    (`experiments/figures/ablation_paired.png`) plotted the original Phase 2c eval-batch-32
    result (Δτ −0.0117, p = .22), while §5 ¶2 reports batch 8 (Δτ −0.0235, p = .07, O8). A reader
    would have seen two different numbers for one comparison. FIG-R1
    (`phase6_xla_default_discovery.png`) carried the titles "architecture matters" and "the
    effect is real", which is the framing rejected by option A (Oct 10), and did not say lr 1e-3.
  - **Fixed:** both figures are now generated in `scripts/paper_figures.py` from the
    eval-batch-8 JSONs (`eval_batch_sensitivity_xla_{default,random}.json`), with asserts
    against the Desk's numbers: `paper/figures/fig_r1_xla_default_staircase.{pdf,png}` (no titles,
    labelled "xla:default, shared lr 1e-3", decomposition panel left to the appendix) and
    `fig_r2_xla_random_paired.{pdf,png}`. The rest of the script's outputs were unchanged by the rerun.
  - **Dropped:** the old FIG-R2's "TPUGraphs paper baseline (0.34)" line had no source we could
    confirm, and it differs from the 0.12 the Desk quotes for xla:default. It must not come back
    without a citation.
  - **Open (the first author's call):** FIG-R1 still shows only the lr-1e-3 result. Option: add a dashed line
    for flat at its tuned LR (τ 0.2255) so the figure shows the gap closing on its own.

- **2026-10-12 — O15: scoring sheet ready; the first author scores Oct 19–22.**
  - **Working sheet:** Claude Doc "Axis 3 scoring sheet — Frame B (10 papers)"
    (https://claude.ai/code/artifact/66007727-813a-47d6-9fb1-10a96b0dd6e2). It has one table per
    paper with Qualifies and Score dropdowns, plus a summary table. Its version history stands in
    for git timestamps while scoring.
  - **Repo copy:** `experiments/axis3_recent_coverage_scores.md` was committed blank (1b2e337).
    Final results are copied there and committed by Oct 23.
  - **Done before scoring:** all 10 PDFs obtained. Claude has not read, pre-screened or
    pre-filled any of the 10 papers.
- **2026-10-12 — O15: Axis 3 recent-work declaration FROZEN (e12c51a).**
  `experiments/axis3_recent_coverage_declaration.md`, approved by the first author. Frozen before any of the
  10 Frame B papers was opened; all 10 PDFs were downloaded unopened beforehand, and all were
  obtained.
  - **Pre-freeze checks:** A19's three quotes verified verbatim against the full arXiv HTML, so
    the model-only scope rule matches precedent. The "significantly" rule applies the existing
    fail anchor. Supplement scope is main PDF plus in-PDF appendix only, as for the 20.
  - **§3a (new rules) is empty.** No second rater.
  - **Next:** the first author scores B1 to B10 in order, Oct 19–22, Axis 3 only; results by Oct 23. If the
    week slips, fall back to Option A.
- **2026-10-12 — O15: coverage assessment; Option B chosen; Axis 3 recent-work declaration DRAFTED (not frozen).**
  `experiments/audit_coverage_assessment_2026-10-12.md` (747af5c, read-only) re-checked the registered
  figures (Frame A 8/30, Frame B 0/10, frame retrieves 10/20 corpus papers; all confirmed) and found
  that the concern was mis-stated. 9 of the 20 corpus papers are from 2024–26. The real gap is a
  sub-area: 7 of Frame B's 10 are TVM-style tensor-program tuning papers in IEEE/ACM venues, and
  Frame B's citation counts are only 2–16. **the first author chose Option B:** an Axis-3-only check of the 10
  Frame B papers, human-scored by the first author, reported separately and never merged into the 20-paper
  numbers. The draft is `experiments/axis3_recent_coverage_declaration_DRAFT.md`.
  - **Scope rule:** headline claim = the first model-vs-model comparison. Anything that changes
    more than the model counts as whole-system and gives N/A. This is copied from the scoping of
    A6, A10, A19 and the Ansor exclusion.
  - **A19 check:** checked against the rule (AI fetch of arXiv HTML), it qualifies: "drop-in scoring
    component inside the existing search loop", same 64-trial budget. The rule matches precedent.
  - **Before freezing (by Oct 17):** the first author confirms the A19 quotes against the PDF; ticks the
    "significantly" rule (applies the existing anchor, or new dated clarification); confirms the
    supplement scope; downloads all 10 PDFs unopened. 5 are closed-access.
  - **Schedule:** scoring Oct 19–22 in list order, results by Oct 23. If the week slips, fall back to
    Option A (disclose only).
  - **Either way:** the post-hoc year split goes in §8, labelled as post-hoc. None of the 9 recent
    papers runs a test, and 7/9 fail 3 or more axes vs 10/11 older ones.
  - **Wording flag:** "recently published" (`phase7_meta_analysis.md:581`) must not reach the
    manuscript.
- **2026-10-11 — Top-K at the tuned LRs: result (declared 6def25d, exploratory).**
  `experiments/topk_slowdown_xla_default_tuned.{md,json}` were computed on Oct 10 at 16:45 by the
  declared command and left uncommitted. Re-run on Oct 11 into scratch: **identical** (max abs diff
  0.0). Flat@1e-4 vs hardware@1e-4 on xla:default, slowdown@1: 1.0176 ± 0.0091 vs 1.0042 ± 0.0040,
  Δ −0.0134 (p = .079); @5 −0.0055 (p = .176); @10 −0.0008 (p = .203); @100 −0.0002 (p = .429).
  **No significant Top-K difference at the tuned LRs.** O9's lr-1e-3 Top-K win (1.152 → 1.009,
  p ≤ 1e-4) is scoped to "at a shared lr of 1e-3", consistent with 6f's τ result. Caveat as
  declared: LR* was chosen on τ over the same validation graphs.
- **2026-10-11 — 8e frozen + launched; R0 frozen + computing; R4 declared + queued; O19, O20 opened.**
  8e frozen (44267c7, the first author sign-off), queue k (b13fcee). Amendment 1 to 8e and R0 (5defe35), made
  before any data: CONV Top-1 at eval batch 32 as 8c (O19: 8c/8d's batch was never stated), with
  batch 8 as sensitivity. R0 computing (`scripts/r0_conv_top1_eval.py`). Leave-one-graph-out on 8c
  (post hoc): the effect survives every single-graph drop. R4 declared (frozen on commit; the first author
  approved queueing, design reviewable until it starts ~Oct 15): queue l runs any 8e boundary step
  first (`scripts/phase8e_lr_star.py`), then R4. The untracked tuned Top-K files trace to an Oct 10
  session that ran the declared command (6def25d) but never committed; a re-run to confirm they
  reproduce is in progress.
- **2026-10-11 — 6d §10.5 evaluated (amended, exploratory).** `scripts/phase6d_amended_evaluate.py` →
  `experiments/phase6d_amended_results.{md,json}`. All 13 runs done before the Oct 23 cutoff, LR
  confirmed, no collapses (val_n_graphs 20). A-P1 hardware@1e-4 − flat@1e-4 = −0.0051 (p = .527,
  boot CI [−0.019, +0.006], MDE 0.027) → §7 "Δτ < 0 (any p)": A-P1 and A-P2 fail. The confound check
  (hardware@1e-4 − flat@1e-3 = +0.089, p = .0195) shows the 6e pattern again: flat@1e-3 is
  handicapped here too (0.3192 vs 0.4138 at 1e-4; post hoc). O10 updated. Option A: no exception; the
  null holds on all four collections (nlp:default amended). Desk §5 ¶10 slot + scope lines updated.
- **2026-10-11 — Framing audit (read-only) + external review.** `experiments/framing_audit_2026-10-11.md`.
  (1) **O18 opened:** 8c baseline not LR-tuned; can't tell whether part of 1.42 → 1.12 is an LR
  artifact → 8e drafted (`phase8e_lr_robustness_preregistration.md`, NOT frozen). (2) **T3 counting
  bug:** the model-free scan counted 16 test graphs that have no runtimes; correct n = 95 / 49 (not
  105 / 55), 2.77×, MWU p = 1.5e-7; ρ +0.52 tie-corrected (not +0.54). The audit first suggested the
  de-duplicated 4.35× as the headline; external review rejected that as post-hoc number-picking.
  The headline stays 2.77×, and 4.35× / per-collection are labelled robustness checks. (3) Fixed:
  channel BH-p .0032 (outline said .004); the power addendum's "20/20 no seed variance" → 16 F / 4 ∂;
  Desk §7 ¶7 "81% favours representation" removed; "MXU tiling geometry" → channel-padding
  utilisation; "Level 2 is NOT exposed" retracted (its 0.288 was the Phase 2c flat, not the 8c
  baseline's 0.2311). (4) O17 reworded to "not tested at matched hardware". (5) R0 eval-only
  declaration drafted (`r0_conv_top1_structure_declaration.md`). Not blind: Phase 5's cache already
  held flat vs hardware CONV Top-1 on xla:random (hw ≈ flat or worse), seen while drafting. (6)
  Novelty draft marked SUPERSEDED. Not changed (the first author's prose): the `\section` title of
  `paper/sections/05_result_distribution.tex` still says "Distribution-Dependent".
- **2026-10-10 — Finding 1 framing DECIDED: option A ("structure vs information").** the first author chose
  A over B (trainability observation, post hoc, n = 2) and C (demote L1). L1 now reads: added
  architectural structure (hierarchy, contention edges) beats a matched flat model on none of the
  three collections with results; only xla:default is LR-tuned on both arms (+0.002, p = .88,
  MDE 0.037), xla:random / nlp:random are at one shared LR (nlp:random hw marginally worse, p = .0504); the apparent +0.165 at a shared lr of
  1e-3 was a flat-model LR handicap caught by the pre-registered 6e check, reported first as a
  failure and then as a worked instance of audit Axis 5's BASELINE clause (rubric name: "Effect
  Size vs Noise & Baseline Strength"; passes the noise half, fails the baseline half). Do not
  claim the audit shows practice "cannot tell structure from information" (it doesn't measure that;
  external-review catch, 2026-10-10). Guardrails: "no detectable gain (MDE …)",
  never "structure doesn't help"; xla:random / nlp:random were compared at one shared LR only.
  §5: ¶7 (6b decomposition, O9 Top-K) and ¶9 (inception_v3) → appendix scoped to lr 1e-3; ¶8 (O11
  "why hierarchy helps") cut. L2 not exposed (8c trained on xla:random, flat@1e-3 τ 0.288). Pending:
  B1 nlp:default (6d §10.5, Oct 11) may add one labelled exploratory exception; Top-K at the tuned
  LRs (declared 6def25d) not yet computed. Desk claim lines + paper_outline updated; the first author rewrites
  the intro / abstract / §10 prose.
- **2026-10-10 — Kaggle test-set scoring: feasibility check only (nothing submitted).**
  Local test npz files carry `config_runtime` = all zeros (checked all 4 layout collections; 1000
  configs per graph; xla:default and xla:random have 8 test graphs, nlp 17 each), so test labels
  exist only on Kaggle. The competition (`predict-ai-model-runtime`) closed 2023-11-17; the first author's
  account is registered for it with no submissions. One submission file covers all 5 parts
  (tile + 4 layout collections) and returns a single public/private score. Isolating one
  collection would need differencing between submissions that hold the other parts fixed (e.g.
  10 submissions = 5 seeds × flat/hardware for xla:default). **Not verified** (pages are
  JavaScript-rendered, so neither could be read): whether late submissions are still scored, and
  how exactly the 5 parts combine into the score. Before any use: the first author checks late-submission
  availability on the competition page; any submission is made under his account with his
  explicit OK; the analysis is declared and committed before the first score is read.
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
