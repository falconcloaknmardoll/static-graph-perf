# Phase 7 — Literature Meta-Analysis (evaluation rigor audit)

> **STATUS: STARTED 2026-06-29, updated 2026-08-25. 20 / ~30 papers AI-prescreened. Corpus has
> crossed the ≥20 threshold for a headline percentage. AI-vs-AI second-rater pilot done (84% raw
> agreement, κ≈0.67) AND a source-grounded human-directed verification pass on the same 5-paper
> (25%) sample is now also done, across all 5 axes (§D item 2) — 3 corrections found across 25
> axis-scores checked (12%), all moving scores stricter. Headline stats below (§D item 3) are
> now citable, on the methodology that ≥20% of the corpus received full human verification and
> the remaining 80% is AI-prescreened only (limitation stated explicitly in §D item 2).**
> This operationalizes `meta_analysis_rubric.md` (5 axes, P/∂/F). The §6 "money quote"
> ("X% of architectural claims would not survive a parameter-matched multi-seed paired
> test") is DELIBERATELY left blank until enough papers are scored from their actual text.

## Method & anti-fabrication guardrail
Auditing papers is the highest hallucination-risk task in this project, so:
1. **Source-grounded only.** A paper is scored only from its *actual text*; every cell is
   backed by a direct quote or an explicit "NOT STATED". No scoring from memory/abstracts.
2. **Fetch workflow.** Recent papers with arXiv **HTML** can be machine-pre-screened
   (WebFetch with a quote-or-"NOT STATED" prompt). **PDF-only papers (older) cannot be
   reliably fetched** — WebFetch returns binary; these are flagged `human-read`.
3. **Even HTML extraction needs a human spot-check** — the fetch uses a small model. Each
   score below is `AI-prescreened`; a human confirming the quotes promotes it to `verified`.
4. **Conservative scoring.** When ambiguous, score the weaker category (rubric rule).
5. **Aggregate-only reporting; no singling out.** Findings are reported as corpus-level
   statistics ("X of 20 papers fail axis 3"), never as named indictments of individual
   papers or authors in the prose. Per-paper scores stay in §A as the auditable evidence
   trail; the write-up argues about *a field-wide evaluation norm*, not about anyone's
   competence. Rationale, on the merits: (a) the claim being made IS distributional — a
   single paper's axis-3 failure is uninformative, the 100% rate is the finding; (b) n=1
   callouts invite ad hominem reading and are the standard failure mode of rigor
   meta-analyses; (c) 80% of the corpus is AI-prescreened only (guardrail 3), so
   per-paper assertions are not yet individually defensible at the confidence a named
   callout implies. Tone is deferential throughout — this work is only measurable
   *because* these papers built the benchmarks and baselines.
6. **Rubric applied identically to every paper.** No paper is scored more leniently or
   more strictly on account of its authors, venue, or the auditors' relationship to
   them. Counter-examples are sought and reported as actively as failures (cf. PerfVec
   A20, HLS-GNN A18).

> **Authorship-neutrality statement.** The rubric was applied identically to every paper
> in the corpus. No score was conditioned on a paper's authors, their seniority, their
> institution, or its venue — and the per-paper scores in §A are released in full so that
> any reader can check this rather than take it on trust. Where the corpus contains work by
> prominent groups, those papers are scored by the same rules and with the same
> conservative tie-breaking as everything else, and counter-examples are reported as
> actively as failures (cf. PerfVec A20, HLS-GNN A18).


## Selection protocol — written RETROSPECTIVELY on 2026-09-27

> This section was written **after** all 20 papers were scored. It records how the corpus was
> actually assembled; it is not a protocol that was followed. Answers objection O15 in
> `DEFENSIBILITY.md`.

- **Scope rule (fixed before scoring, `meta_analysis_rubric.md`):** a paper is in scope if it
  makes a comparative architectural claim of the form "architecture/feature X improves
  ranking/accuracy over baseline Y" for a learned model that predicts the performance, cost,
  runtime or throughput of programs, kernels or ML models on hardware. Papers without such a
  claim are out of scope. Each §A entry carries its own in-scope note.
- **How the 20 were chosen (purposive/snowball, not systematic):**
  - *Pilot (A1–A2, from 2026-06-29):* TPUGraphs, the benchmark this project uses, and one
    recent architecture paper on it.
  - *Round 1 (A3–A10, completed by 2026-08-17):* the established lineage of learned cost and
    performance models (Ithemal, GRANITE, ProGraML, TLP, PerfSAGE, NeuSight, Kaufman et al.,
    AutoTVM), selected by the authors' knowledge of the area.
  - *Round 2 (A11–A20, completed by 2026-08-17):* found by web searches run by background
    agents. **The search terms, sources and dates of these searches were not logged.** Selection
    was also steered toward finding counter-examples (guardrail 6), which biases the sample
    toward papers that *pass* axes, i.e. conservatively for the headline.
- **Years covered:** 2018–2026. **Venues:** mixed (ML, systems, architecture, EDA, arXiv-only).
- **Excluded or not scored, with reasons (complete list):**

  | Candidate | Reason |
  |---|---|
  | Ansor | Read; ruled out of scope (no comparative architectural claim about a cost model). |
  | Halide autoscheduler (Adams et al. 2019) | PDF-only, not reachable by the fetch workflow; never scored. |
  | AUTOCOMP (2505.18574) | Provisionally out of scope (LLM-driven search, no model training); not confirmed by a full read. |
  | DiffHLS (2604.09240), Intelligent4DSE (2504.19649), TCL (2604.12891), GPU Forecasters (2605.31464), "Can LLMs Predict Parallel Code Performance?" (2505.03988) | Found by search snippet only; not fetched or scored before the n=20 freeze. |
  | Kaggle "predict-ai-model-runtime" writeups | Not peer-reviewed or on arXiv; scope decision never made; not scored. |

- **Consequence for claims:** the corpus is a **purposive sample of 20 canonical and recent
  papers**, not a systematic sample of the field. Claims are worded accordingly ("in a purposive
  sample of 20 …"). How representative it is gets checked by the coverage check below.

## Coverage check — DECLARED 2026-09-27, before the search was run

> Committed before any query was sent. Purpose: measure how much of a reproducible,
> citation-ranked frame the purposive corpus already covers. **No new paper is scored.** The
> corpus stays frozen at n=20.

- **Source:** Semantic Scholar Graph API relevance search (`/graph/v1/paper/search`), top 100
  results per query, year range 2018–2026. Raw responses are saved to `experiments/phase7_coverage/`
  with the retrieval date (citation counts drift over time).
- **Queries (fixed; one per sub-area of the scope rule):**
  1. `learned cost model tensor program`
  2. `learned performance model`
  3. `graph neural network performance prediction`
  4. `basic block throughput prediction`
  5. `deep learning model latency prediction`
  6. `high-level synthesis performance prediction graph neural network`

  *Disclosed:* the sub-areas were chosen knowing the corpus, so the frame tests coverage
  *within* these sub-areas. It does not test whether a whole sub-area is missing.
- **Frames:** results are unioned, deduplicated, and sorted by citation count, descending.
  - **Frame A (canonical):** the **30 most-cited in-scope papers**.
  - **Frame B (recent):** the **10 most-cited in-scope papers from 2024–2026**, because
    citation ranking disadvantages recent work.
- **Screening:** title + abstract, in citation order, against the scope rule above. Screened
  by Claude, with a one-line reason logged for **every** paper screened, and flagged for the
  author's spot-check. **Ambiguous abstracts are screened IN.** This enlarges the frame with
  papers the corpus may not cover, so it can only lower measured coverage.
- **Match rule:** a frame paper is covered if the same work (same arXiv ID or same title) is in §A.
- **Reported, whatever the number:** coverage X/30 (Frame A) and X/10 (Frame B), plus the
  list of uncovered in-scope papers. The numbers go into the paper verbatim, and the uncovered
  list is disclosed as not audited.

## §A — Verified-pilot scorings (quote-grounded)
Legend: P=pass ∂=partial F=fail. Confidence = AI-prescreened (needs human quote-check).

### A1. TPUGraphs (Phothilimthana et al., NeurIPS D&B 2023) — arXiv 2308.13490 [HTML]
*In-scope note:* primarily a dataset paper, but makes comparative GNN claims (GraphSAGE vs
GCN, model variants) — those are what we score. It is the benchmark our project builds on,
so auditing it is calibration, not an attack.

| Axis | Score | Evidence (quoted from the paper via HTML fetch) |
|---|---|---|
| 1 Seeds/variance | **F** | Layout: "we train the model once with only a few set of hyperparameters"; Tile: "we train each model variant three times" but **no std/CI reported**. Headline layout = single run. |
| 2 Param-matched | **F** | NOT STATED — no capacity/param control between compared variants. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — no t-test/Wilcoxon/bootstrap. |
| 4 Eval-set integrity | **∂** | "We split the data using 80-10-10 ratio by graphs"; reports test τ (held-out test exists ✓) but few eval graphs per collection + "report final metrics only on 1,000 configurations", no power caveat. |
| 5 Effect vs noise | **F** | "GraphSAGE vs GCN … comparable" with no numbers/variance → gain-vs-noise not assessable. |
*Fails ≥3 of 5 axes (1,2,3,5).*

### A2. Cross-attention "TGraph" (tensor-compiler GNN, 2024) — arXiv 2405.16623 [HTML]
*In-scope:* central claim is architectural (configuration cross-attention beats baselines).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **∂** | Reports dispersion ("confidence ranges … 1-sigma", "0.6840 ± 0.0110") — better than most — but via "K-fold cross-validation with K=20" and "pick the top-4 folds by the validation score" (fold selection, not independent seeds). |
| 2 Param-matched | **F** | NOT STATED — "does not report model sizes, parameter counts, or capacity comparisons". |
| 3 Significance test | **F** → **∂** (human verification 2026-10-05) | No hypothesis test, p-value or CI, but variance IS reported for the headline's components: Table 2 gives ±1-sigma for every TGraph cell feeding the layout mean — 0.6840 ± 0.0110 (xla:random), 0.4785 ± 0.0031 (xla:default), 0.9713 ± 0.0008 (nlp:random), 0.5628 ± 0.0027 (nlp:default); §2.5.4: "the confidence ranges are reported as 1-sigma." The headline mean row (0.674 vs 0.610) carries no propagated uncertainty and the Xu 2023 baseline has none. Matches the rubric's ∂ anchor ("reports variance but no test"). |
| 4 Eval-set integrity | **F** | "TpuGraphs … does not provide public test data annotations. Hence, we report the cross-validation score" on "merged train/validation" AND "pick the top-4 folds by the validation score" → selection leakage, no held-out test. |
| 5 Effect vs noise | **∂** | TGraph τ=0.674 vs Xu-2023 0.610 (Δ≈0.064) with ±~0.011 ranges → gain exceeds its own σ, but no param-match so not attributable to the idea; baseline reasonable. |
*Fails 2 of 5 outright (2, 4) after the 2026-10-05 Axis 3 correction; has dispersion (Axes 1, 3 ∂) but a leaky eval (Axis 4 F).*

### A3. GRANITE (Sýkora et al., Google, basic-block throughput GNN) — arXiv 2210.03894 [PDF, read via ar5iv HTML render]
*In-scope:* central claim is architectural — Granite (GNN) beats Ithemal (LSTM) and an
enhanced "Ithemal+" baseline on basic-block throughput estimation.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. Headline is a single point estimate: "Our results establish a new state-of-the-art for basic block performance estimation with an average test error of 6.9%." No seed count, no std/CI anywhere in the results. |
| 2 Param-matched | **F** | NOT STATED. Table 4 gives Granite's own hyperparameters (e.g. "Node Embedding Size: 256") but no total parameter count for Granite, Ithemal, or Ithemal+, and no capacity-matched comparison; paper only says "we re-implemented the Ithemal model using the same version of TensorFlow." |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — no t-test/Wilcoxon/bootstrap CI anywhere; improvement asserted from raw MAPE numbers: "Granite outperforms Ithemal across all the microarchitectures by at least 1.67%, and by 1.93% on average." |
| 4 Eval-set integrity | **P** | "we randomly split each dataset into a training part comprising 83% of blocks and a test part containing 17% of blocks. We use the *same* split of the data set in all experiments." Test set is large (>300K blocks on BHive, >1.4M on the Ithemal-derived set) and selection/tuning is done on a separate slice: "We further split the training data into training (98%) and validation (2%)." Held-out test, adequate size, selection kept separate. |
| 5 Effect vs noise | **∂** | Gain vs Ithemal is consistent across chips ("outperforms Ithemal across all the microarchitectures by at least 1.67%") and Ithemal is the genuine prior SOTA (not a strawman), but the margin over the *enhanced* baseline is much smaller — Ithemal+ 7.74% vs Granite 6.91% avg MAPE (~0.83 pts) — and with zero variance reported (Axis 1 = F) there is no way to check either gap against a noise floor. Conservative call: partial, not pass. |
*Fails 3 of 5 axes (1, 2, 3); notably the only paper so far to score full P on Axis 4.*

### A4. Ithemal (Mendis et al., MIT, basic-block throughput LSTM) — arXiv 1808.07412 [PDF, read via ar5iv HTML render]
*In-scope:* central claim is architectural/representational — a learned LSTM model beats
hand-built analytical simulators (IACA, llvm-mca) on throughput estimation accuracy.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. "We randomly assign 80% of the collected blocks to the train set and 20% to the test set" — no mention of repeated runs, seeds, std, or CI anywhere in the reported error/correlation tables. |
| 2 Param-matched | **F** | NOT STATED. No parameter counts given for Ithemal, and the baselines (IACA, llvm-mca) are hand-written analytical tools with no comparable "capacity" concept, so a capacity-matched comparison is neither performed nor really well-posed here — but the rubric still requires an explicit statement/control for the axis to pass, and none exists. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED. Table 3 reports point-estimate error/correlation metrics with no confidence intervals or hypothesis tests. |
| 4 Eval-set integrity | **P** | "We randomly assign 80% of the collected blocks to the train set and 20% to the test set" on a dataset of "1,416,473 unique basic blocks" (Haswell) — large, clearly held-out test split, standard random assignment, no stated selection leakage. |
| 5 Effect vs noise | **∂** | Effect size is large and the baselines are legitimate, widely-used industrial tools rather than strawmen: Haswell error 0.089 (Ithemal) vs 0.200 (llvm-mca) and 0.209 (IACA), Spearman correlation 0.890→0.960 — roughly a 55% relative error reduction. This is a big gap unlikely to be pure noise, but since Axis 1 is F (no variance/seed data at all), the paper itself gives no way to formally rule out noise, so we do not award full P without that check. |
*Fails 3 of 5 axes (1, 2, 3); like GRANITE, clean on eval-set integrity (Axis 4 = P) and has
a large, plausibly-real effect (Axis 5 = ∂) despite the missing statistics.*

### A5. ProGraML (Cummins, Fisches, Ben-Nun, Hoefler, Leather; ICML 2021) — arXiv 2003.10536 [HTML, ar5iv]
*In-scope:* central claim is architectural/representational — the ProGraML graph
representation + GNN beats prior representations (DeepTune, DeepTuneIR, NCC/inst2vec, an
LSTM baseline) across three case studies (compiler analyses, device mapping, algorithm
classification).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. Compiler-analyses results (Table 3) are single point estimates (e.g. "0.940 F₁"); device-mapping (Table 5) uses "10-fold cross-validation with rotating 80/10/10 splits" but reports only point accuracy — no std/CI across folds or seeds anywhere. |
| 2 Param-matched | **F** | Only partially disclosed and not compared: "the Transformer model has 5.6 million trainable parameters, around 1.7 million of which are in the embedding layer" — but no parameter counts given for DeepTune, DeepTuneIR, NCC, or the LSTM baseline, so no capacity match is possible or attempted. |
| 3 Significance test | **F** | NOT STATED — no p-values, t-tests, or CIs reported for any of the three case studies' headline gaps. |
| 4 Eval-set integrity | **∂** | Mixed across case studies: Case Study A (compiler analyses) uses a clean "3:1:1 ratio for training, validation, and test instances" over "1M labeled graphs" (solid); Case Study C (algorithm classification) holds out "approximately 10,000 files... as a development and test set" from 240k samples (solid); but Case Study B (device mapping) has only "680 labeled instances derived from the 256 unique kernels" — small and non-independent (multiple instances per kernel) despite the 10-fold rotation, so the weakest case study pulls the axis to partial rather than full pass. |
| 5 Effect vs noise | **∂** | Mixed: the compiler-analyses gap looks inflated by an incapable baseline — ProGraML "94.0 F₁" vs DeepTuneIR "0.273 F₁", but DeepTuneIR "cannot perform per-variable classification required for Liveness analysis" (i.e. task-incompatible strawman, near-F territory). The other two case studies use a fairer, competitive baseline (NCC) with modest, plausible gaps — device mapping 86.6% vs NCC 80.3% (+6.3 pts), algorithm classification 3.33% vs 5.17% error (−35.6% relative) — but still with zero variance reported to check against noise. |
*Fails 3 of 5 axes (1, 2, 3); flagged because one of its three headline comparisons
(DeepTuneIR) looks like a baseline that could not structurally perform the task, which is a
sharper version of the "strawman baseline" failure mode than most papers in this queue.*

### A6. TLP (Zhai et al., deep-learning cost model for tensor program tuning vs Ansor's XGBoost cost model) — arXiv 2211.03578 [HTML, ar5iv]
*In-scope:* central claim is architectural — a learned (Transformer/MLP-style) cost model
beats Ansor's default XGBoost cost model and the TenSet MLP cost model on tuning search
efficiency.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. Table 5 reports single top-k scores per platform; no error bars, std, or repeated-run reporting anywhere. |
| 2 Param-matched | **F** | NOT STATED. Architecture is described ("Shallow linear layers in the model upsample the embedding size from 22 to 256"; "using one layer of the self-attention module is enough") but no parameter counts or training-budget comparison is given for TLP vs. the TenSet MLP vs. Ansor's XGBoost model. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — no statistical test anywhere in the results section. |
| 4 Eval-set integrity | **P** | Clean, named held-out test networks: "we hold out a test set that consists of five networks, ResNet-50, MobileNet-V2, ResNext-50, BERT-tiny, and BERT-base," with the rest split "training and validation sets at a ratio of 9:1" over the large TenSet corpus ("8.65 million tensor programs"). Held-out networks unseen in training/validation — solid separation. |
| 5 Effect vs noise | **∂** | Headline claim is large — "TLP can speed up the average search time by 9.1× and 3.0× on CPU and GPU workloads, respectively, compared to the state-of-the-art implementation" — but the paper itself flags a quality caveat that tempers a clean "TLP is just better" reading: "There is still a big gap between the performance of Ansor tuning 2000 times and the performance of TLP/MTL-TLP tuning the minimum times." I.e. the win is on search-time-to-a-given-quality, not on quality at matched budget, and that tradeoff is stated but not quantified with variance. |
*Fails 3 of 5 axes (1, 2, 3); a genuine counter-example on eval-set integrity (Axis 4 = P,
named held-out networks) but the self-reported quality-gap caveat keeps Axis 5 at partial
rather than full pass.*

### A7. PerfSAGE (GNN-based inference-performance predictor for edge DL models) — arXiv 2301.10999 [HTML, ar5iv]
*In-scope:* central claim is architectural — PerfSAGE (GNN) beats prior latency predictors
(Eagle, nn-Meter) across multiple TFLite model search spaces and hardware targets.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. No standard deviations, confidence intervals, or error bars appear anywhere in the results tables; MAPE is reported as single numbers per target. |
| 2 Param-matched | **F** | NOT STATED — no parameter counts for PerfSAGE, Eagle, or nn-Meter are given, and no capacity control between them is discussed. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — only point-estimate MAPE comparisons, no hypothesis test. |
| 4 Eval-set integrity | **P** | "The sampled TFLite models are split into 80% training data and 20% test data" over "134,912 DNNs randomly sampled from four task search spaces" — large, clearly held-out test split. |
| 5 Effect vs noise | **F** | The headline gap is enormous — PerfSAGE "<5% [MAPE] across all targets" vs Eagle's "41.138%" on CNN-CIFAR10 CPU latency — but the paper itself concedes the baseline is out of its depth: "Eagle is primarily designed for NASBench201, a well defined design space with detailed architecture rules. Thus, it struggles with a more diverse collection of models." This is a self-acknowledged mismatched/strawman baseline comparison, the exact Axis-5 F anchor. |
*Fails 4 of 5 axes (1, 2, 3, 5); the one paper so far whose own text concedes the baseline
comparison is structurally unfair, which is a stronger version of the thesis's failure mode
than papers that merely omit the check.*

### A8. NeuSight (GPU performance forecasting via tile-decomposition ML, 2024) — arXiv 2407.13853 [HTML, ar5iv]
*In-scope:* central claim is architectural — a tile-decomposed ML predictor beats prior
end-to-end regression/MLP predictors (Habitat, "Li et al.") on unseen-GPU latency forecasting.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED for model variance/seeds. "Latency is measured by running each operator 25 times and averaging the results" is a *measurement*-noise control, not a training-seed/variance report; no std/CI for the learned model's predictions across runs. |
| 2 Param-matched | **F** | NOT STATED / not matched. Architecture given for NeuSight's own MLP ("Each MLP comprises 8 hidden layers, each with 512 hidden units") but the Habitat baseline is explicitly *not* capacity-matched — the paper reports diverging from Habitat's original size: "we train the MLP following the methodology presented in their paper, but with the larger MLP for higher accuracy," with no total parameter counts given for either model. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — only point-estimate percentage errors reported. Human check: Fig. 7 errors are point averages per model and GPU; the OOD average (8.1%) and maximum (28.2%) are a range across workloads, not run-to-run variance. |
| 4 Eval-set integrity | **P** | Explicit, strong held-out design: trained on "5 training set GPUs," tested on a GPU variant never trained on ("A100-40GB variant was used for training but the A100-80GB variant... was used as the testset") plus genuinely out-of-distribution GPUs ("H100, L4"), and models "trained only on a subset of GPT3 and OPT model dimensions" then tested beyond that subset. One ambiguity noted: "We reserve 20% of this sample dataset for validation" but it is unclear whether that validation informed hyperparameter choices in a way that could leak into the OOD test claims — not enough to drop below P given how explicit the GPU/model holdout otherwise is. |
| 5 Effect vs noise | **∂** | Very large claimed gains against a genuine (not obviously strawman) prior-SOTA baseline — in-distribution: "NeuSight shows a percentage error of only 9.7% for inference and 7.3% for training" vs "Habitat shows 220.9% and 725.8%, and Li et al. shows 61.2% and 58.3%"; OOD headline: reduces GPT3-on-H100 error "from 121.4% and 30.8% to 2.3%." The paper made a stated effort at a fair comparison ("For fair comparisons, we train the baselines with the up-to-date training dataset as NeuSight"), and the gap is qualitatively far larger than any plausible seed noise — but with zero variance/seed data reported (Axis 1 = F), there is no formal way to confirm this against a noise floor, so conservative scoring keeps this at partial rather than full pass. |
*Fails 3 of 5 axes (1, 2, 3); like NeuSight's TenSet/TLP cousins, clean on eval-set integrity
(Axis 4 = P, one of the more rigorous held-out designs seen so far — explicit unseen-GPU and
unseen-model splits) with a very large, plausibly-real effect (Axis 5 = ∂).*

### A9. "A Learned Performance Model for Tensor Processing Units" (Kaufman, Phothilimthana, Zhou, Mendis, Roy, Sabne, Burrows; Google, MLSys 2021) — arXiv 2008.01040 [HTML, ar5iv]
*In-scope:* central claim is architectural — a learned GNN-based TPU cost model vs a
heavily hand-tuned analytical performance model, on tile-size selection and operator fusion
tasks, plus an internal GraphSAGE-vs-GAT architecture ablation. **Flagged as a candidate
counter-example** — see notes below; it is not a clean multi-P paper, but it is noticeably
more careful than every other paper scored this session.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **∂** | Headline cost-model accuracy (Table 2, e.g. "3.7% mean error") has NOT STATED seeds/variance — "we did a hyperparameter search...and selected the best-performing model for each task on the validation set," no repeats mentioned. But the fusion-autotuner result *does* report a dispersion: "We run the autotuner on each program 10 times and report the best speedup found... Solid bars show the median best speedup; error bars range from min to max." Partial credit for that secondary result, not the headline comparison — matches the rubric's "seeds for some results but not the headline comparison" ∂ anchor exactly. |
| 2 Param-matched | **∂** | No baseline is a neural net (it's a hand-tuned analytical model), so a param match to it isn't well-posed, but for the paper's *internal* architecture ablation it explicitly claims rough parity: GraphSAGE uses "roughly the same number of learnable parameters compared to GAT" — a real but qualitative/unverified claim (no exact counts given for either), which is more than most papers in this queue attempt. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — no t-test/Wilcoxon/bootstrap CI anywhere; comparisons are point estimates ("3.7% mean error" vs "6.1% mean error"). |
| 4 Eval-set integrity | **∂** | Genuinely rigorous *design* — two separate held-out protocols are tested for robustness: a random split and a harder manual split "chosen by hand to minimize the subjective similarity" between train and test programs, both with test kept separate from training and tuning done on a separate validation slice. But the random-split test set is small: "93" train / "8" validation / "8" test *programs* (the kernel counts within them are large — "1.4M" test kernels — but only 8 held-out programs), which is squarely the rubric's "very few graphs (single digits)" ∂ anchor despite the otherwise careful design. |
| 5 Effect vs noise | **∂** | Not a uniform win — and that is the notable part. Fusion task: learned "4.5 MAPE" clearly beats analytical "31.1 MAPE" (holds up on the manual split too: "6.2" vs "18.1"). But tile-size task on the harder manual split, **the baseline wins**: learned "6.3%" vs analytical "2.3%" mean error — i.e. the paper honestly reports its own method losing to the hand-tuned baseline under the more realistic split, rather than only showing the split where it wins. Baseline is explicitly a strong, "heavily-optimized analytical performance model," not a strawman. |
*Fails only 1 of 5 axes outright (Axis 3); the rest are ∂ rather than F — this is the least-bad
scoring profile of the papers audited so far. The standout finding is Axis 5: this is the only
paper in the corpus so far that reports a split/condition where its own proposed model loses to
the baseline, rather than cherry-picking the favorable comparison. That candor (plus running two
independent held-out protocols, plus reporting min/max dispersion for at least one result) makes
it the strongest counter-example candidate found this session, even though it still lacks
seeds/CI on its headline number and has no significance test at all.*

### A10. AutoTVM (Chen, Zheng, Yan, Jiang, Moreau, Ceze, Guestrin, Krishnamurthy; NeurIPS 2018) — arXiv 1805.08166 [HTML, ar5iv]
*In-scope:* scored specifically for its learned-cost-model architectural claims (GBT vs
TreeGRU cost models, transfer learning across workloads, vs "black-box" search baselines) —
not the paper's broader end-to-end systems claims against cuDNN/TensorFlow, which are a
different (non-architectural) kind of comparison and are out of the rubric's scope.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. *Corrected 2026-10-05 (human check of the PDF):* Figures 4 and 5 are NOT single curves — every curve has a shaded band (C1, C2, C5, C6), but the bands are never defined in the captions or text (searched: shaded, band, interval, deviation, variance, error bar), and no seed or run count is stated. Score kept F: undefined bands are not reported variance (rubric: when in doubt, score the weaker category). |
| 2 Param-matched | **F** | NOT STATED. "We use XGBoost, which has proven to be a strong feature-based model" and TreeGRU "hidden size of 128" (from the appendix) are the only architecture details given; no XGBoost tree depth/leaf-count or total-parameter figures are reported for either side, so no capacity match is verifiable. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — "both the GBT and TreeGRU models outperformed the black-box methods and found operators that were 2× faster" (§6.1 prose only) with no test. Fig. 4 shows unlabeled shaded bands around each curve; because they are never defined, they are not counted as reported variance (rubric tie-break: weaker category). Unlike A2, which states its ranges are 1-sigma. |
| 4 Eval-set integrity | **∂** | The transfer-learning experiment has a genuine held-out design — "We randomly picked samples from 𝒟′ collected from C1,C2,C3,C4,C5,C6 and used them to form the source domain... We then compared the performance... for target workloads C7,C8,C9" — but the target/test set is only 3 workloads (single digits, no power caveat stated), and the single-operator baseline comparisons (Fig. 10) have "no explicit train/test partition... documented." |
| 5 Effect vs noise | **∂** | The specific cost-model claim ("outperformed the black-box methods and found operators that were 2× faster") is against "black-box methods" (i.e. cost-model-free search), which is a plausible but not clearly matched comparison — no detail on how well-tuned that black-box baseline search was, and no variance to check the 2× figure against. Not scored F since 2× is a large, plausible-real gap and this isn't obviously a strawman, but not P given the missing tuning/variance detail. |
*Fails 3 of 5 axes (1, 2, 3) on its cost-model-architecture claims specifically.*

### A11. TenSet (Zheng, Liu, Shao, Chen, Gonzalez, Stoica, Haj-Ali; NeurIPS D&B 2021) — no arXiv ID, NeurIPS proceedings PDF (datasets-benchmarks-proceedings.neurips.cc)
*In-scope:* central claim is architectural — comparing cost-model architectures (MLP, LSTM,
GRU, GraphSAGE, GCN, GBDT/XGBoost) for tensor-program performance prediction, plus an
MLP+ranking-loss model vs Ansor's default cost model on end-to-end search time.
*Provenance note:* no confirmed arXiv ID exists for this paper (OpenReview id aIfp8kLuvc9,
NeurIPS D&B proceedings PDF only); the official NeurIPS proceedings PDF was fetched via a
text-extraction proxy (r.jina.ai) after direct WebFetch of the PDF returned only binary/garbled
content and the OpenReview page returned a browser-verification wall.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. "No standard deviations, confidence intervals, or error bars are reported in Table 4 (model comparisons across networks)"; the only variance figure found, "The measurement variance is around 0.1 ms," is hardware measurement noise, not model/seed variance. |
| 2 Param-matched | **F** | NOT STATED. Architectures listed by type ("MLP, LSTM, GRU, GraphSAGE, GCN, and GBDT") but "the paper does not report hidden layer sizes, parameter counts, or confirm capacity-matched comparisons." |
| 3 Significance test | **F** | NOT STATED — "No p-values, t-tests, or formal significance tests are reported. Comparisons are purely descriptive (e.g., Table 4 bold formatting)." |
| 4 Eval-set integrity | **∂** | Real held-out design is described — "hold out a ResNet-50 (batch size=1, image size=224) as the test set, and use the rest of the dataset as the training set" (§5.1) and "hold out a test set that consists of five networks... train models... on the training set that consists of the whole dataset excluding all the tasks that appear in the above five networks" (§5.2) — but exact eval-set graph/program counts per experiment are never given ("NOT SPECIFIED for individual experiments"), only the 51.6M-record dataset total. |
| 5 Effect vs noise | **∂** → **F** (human verification, §D) | Table 4 top-k score differences are modest — "MLP+ranking reaching 0.92 vs. alternatives at 0.87–0.89" — with no test to confirm signal vs noise. The 10× search-time speedup claim ("reduces the search time by up to 10×") is versus a legitimate baseline (original Ansor, no pretrained cost model), not a strawman. |
*Fails 3 of 5 axes (1, 2, 3); resolves last round's "no arXiv ID" stall via the official NeurIPS
D&B proceedings PDF rather than a fabricated ID.*

### A12. Polynormer (Deng, Yue, Zhang; Cornell, ICLR 2024) — arXiv 2403.01232, Appendix L TpuGraphs case study, [PDF via r.jina.ai text-extraction proxy]
*In-scope:* central claim is architectural — Polynormer (polynomial-expressive graph
transformer) vs GraphSAGE (the TPUGraphs baseline) on TpuGraphs Layout and Layout+Tile
runtime-ranking accuracy. *Provenance note:* resolves last round's stall — ar5iv/plain WebFetch
of both the HTML and PDF versions truncated before reaching Appendix L three separate times;
the r.jina.ai proxy fetch of the raw PDF finally surfaced the appendix text.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. "Tables 8 and 9 report only point estimates without standard deviations or error bars, and no mention of multiple runs or seeds appears." |
| 2 Param-matched | **F** | NOT STATED. "The paper provides no model size or parameter count comparisons for either model on this task." |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — Appendix L Tables 8–9 (TpuGraphs) are single numbers, no ±, no test. *Scoping:* the paper's main Tables 1–3 (node classification) do report ± std over 10 runs with no test, which would be ∂; F holds only because the audit scores the TpuGraphs comparison (see in-scope note). *Claim-accuracy notes (Axis 5 material, score unchanged):* the text says Polynormer/Polynormer-local beat GraphSAGE "on all collections", but Table 8 has Polynormer-Local below GraphSAGE on XLA-Default (73.7 vs 77.3); the text gives the XLA-Random gain as 10.3 points, Table 8 gives 87.6 − 76.3 = 11.3. |
| 4 Eval-set integrity | **F** | The paper states "we only change the model architecture while keeping all other configurations the same" but "the specific data split proportions are not described anywhere in the paper" for the TpuGraphs experiments — matches the rubric's "never states which split" F anchor exactly, even though TPUGraphs itself has an official split elsewhere. |
| 5 Effect vs noise | **∂** | Large claimed gains — "10.3% accuracy improvement over GraphSAGE... on the XLA-Random collection" (76.3%→87.6%), and on the full Layout+Tile benchmark "31.8% accuracy improvement over the baseline model" (30.1%→61.9%) — clearly non-trivial margins, and GraphSAGE is a legitimate baseline (TPUGraphs' own default), not a strawman. But with zero variance/seed data (Axis 1 = F), there is no formal way to check against noise, so conservative scoring keeps this at partial. |
*Fails 4 of 5 axes (1, 2, 3, 4) — the weakest eval-set-integrity score of the corpus so far
(Axis 4 = F, not just ∂), despite reporting the largest raw accuracy deltas seen in this queue.*

### A13. "Learned Cost Model for Placement on Reconfigurable Dataflow Hardware" (Guha, Jiang, Deng, Zhang, Annamalai; arXiv 2511.01872, 2025) — [PDF via r.jina.ai text-extraction proxy]
*In-scope:* central claim is architectural — an MLP-based learned throughput regressor vs
a "heuristic-based cost model derived from expert knowledge" for place-and-route decisions on
reconfigurable dataflow hardware. *Provenance note:* resolves last round's stall — direct
WebFetch of the PDF returned suspiciously thin/garbled content (falsely reading as "NOT STATED"
across the board); the r.jina.ai proxy fetch surfaced substantially more real text, though some
fields are still genuinely unstated in the paper itself (not an extraction artifact this time).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **∂** | The paper states it used "5-fold cross validation across our experiments," which is a form of multiple-run evaluation, but "provides no information about random seed initialization or reported standard deviations/confidence intervals" — i.e. dispersion across those 5 folds is never reported for the headline numbers. Matches the rubric's ∂ anchor ("multiple runs but no dispersion reported"). |
| 2 Param-matched | **F** | NOT STATED. The model is described only as "a simple 3-layer multi-layer perceptron using ReLU as the nonlinear activation function" with "no specification of layer dimensions, parameter counts, or model size," and no capacity discussion versus the heuristic baseline (which, like Ithemal/GRANITE's analytical baselines, has no comparable "capacity" concept, but the rubric still requires an explicit statement to pass). |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — "no formal statistical hypothesis testing or significance testing procedures." |
| 4 Eval-set integrity | **F** | "specific proportions for train/validation/test are absent"; only the total dataset size is given ("we collect 5878 pairs of PnR decisions and normalized throughputs in total"), with no confirmation selection and reporting splits differ. |
| 5 Effect vs noise | **∂** | Headline claims — "31%-52% more accurate[y]" than baseline (abstract), and in-text "21% improvement in relative error and 34% improvement in Spearman rank correlation" — are sizeable, and the baseline is explicitly described as a real, effortful analytical system ("It takes a large engineering team to develop and refine these heuristics"), not an obvious strawman. But with no variance/seed dispersion reported for these numbers, conservative scoring keeps this at partial. |
*Fails 3 of 5 axes (2, 3, 4); resolves last round's "suspiciously thin extraction" flag — this is
a genuinely under-reported paper on rigor, not an artifact of the earlier failed fetch.*

### A14. Facile (Abel, Sharma, Reineke; IISWC 2023) — arXiv 2310.13212 [HTML]
*In-scope:* central claim is architectural/comparative — Facile, a fast **analytical**
basic-block throughput model, is claimed to match or beat learned models (Ithemal, GRANITE) on
accuracy while being far faster; scored under the rubric's scope rule because the comparison is
against learned-model baselines even though Facile itself is not learned (same situation as
A4's Ithemal-vs-analytical-tools comparison, mirrored in the opposite direction).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED — "No information provided about multiple runs, random seeds, or standard deviation/confidence intervals for the headline accuracy metrics in Table 2." |
| 2 Param-matched | **F** | Facile is analytical (no learned parameters), so a capacity match to it is not well-posed — but per the A4 precedent, the rubric still requires an explicit capacity-control statement to pass, and none is given for the comparison overall. |
| 3 Significance test | **F** | NOT STATED — "No statistical significance tests reported for accuracy comparisons." |
| 4 Eval-set integrity | **∂** | Evaluated on "a modified version of the BHive benchmarks" (BHiveU/BHiveL subsets filtered to fit Facile's modeling assumptions) with "no explicit train/test split mentioned; appears to be single evaluation set" — Facile requires no training, so train/test separation is less applicable, but the modified/filtered benchmark subset (chosen to fit the model's own assumptions) is a mild selection-bias concern. |
| 5 Effect vs noise | **∂** | Facile beats Ithemal by a large, directly-measured margin ("Facile: 0.42% \| Ithemal: 8.28%" MAPE on Rocket Lake BHiveU, and "~33× faster"), matches uiCA closely ("0.42% vs 0.49%"), and claims it "significantly outperforms GRANITE on all microarchitectures" — but that GRANITE comparison is explicitly **not** a controlled head-to-head: "they could not run GRANITE themselves, citing results from its paper," i.e. borrowed numbers under possibly different conditions, the exact kind of uncontrolled cross-paper comparison the rubric flags. The Ithemal/uiCA comparisons look directly run and are large enough to plausibly be real; the GRANITE comparison specifically should not be trusted as apples-to-apples. |
*Fails 3 of 5 axes (1, 2, 3); notable as a rare case where the "learned model" is the losing
baseline rather than the proposed system, and where a cross-paper (non-reproduced) comparison
is explicitly admitted in the text.*

### A15. COGNATE (Sudusinghe, Gerogiannis, Lenadora, Block, Torrellas, Mendis; ICML 2025) — arXiv 2506.00424 [HTML]
*In-scope:* central claim is architectural — a transfer-learning cost model (pretrained on
general-purpose hardware, fine-tuned on emerging accelerators) vs baseline cost models
(WacoNet, WACO+FA, WACO+FM) for sparse tensor program (SpMM/SDDMM) optimization.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED. "The paper reports geomean speedups (e.g., '1.47× (up to 5.46×) for SpMM') but provides no information about number of experimental runs, seeds, standard deviations, confidence intervals, or error bars." |
| 2 Param-matched | **F** | NOT STATED. "The paper does not report the total parameter counts for COGNATE's model or baseline models (WacoNet, WACO+FA, WACO+FM), nor does it discuss whether models were capacity-matched." |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — "No statistical significance tests are mentioned or reported." Fig. 4 is geomean bars without error bars; per-matrix plots (Figs. 5, 13–15) are descriptive. *Claim-accuracy note (Axis 5 material, score unchanged):* the text calls WACO+FM the closest alternative (28.44% margin), but in Fig. 4 "No Transfer" is closer on every operation, e.g. SpMM (SPADE) 1.29× vs COGNATE Top-1 1.40× (≈8.5%). |
| 4 Eval-set integrity | **P** | Explicit held-out design: "715 real-world matrices from the SuiteSparse Matrix Collection, ensuring that none of the evaluation data samples overlapped with the training set" — training data separately described as "1,500 matrices for each hardware platform, with up to 1,000 matrices used for model training... and the remainder was set aside for validation." Clear non-overlapping test set of adequate size. |
| 5 Effect vs noise | **∂** | Headline speedups ("1.47× (up to 5.46×) for SpMM and 1.39× (up to 4.22×) for SDDMM") are large versus WacoNet, a genuine prior-SOTA baseline — but one ablation baseline is notably weak ("WACO+FA... achieving only 1.04×"), a mild "some comparison points are barely-tuned variants" concern, and no variance is reported to check the headline numbers against noise. |
*Fails 3 of 5 axes (1, 2, 3); a genuine counter-example on eval-set integrity (Axis 4 = P, one
of the cleanest non-overlap statements seen in the corpus).*

### A16. Regression Language Models for Code (Akhauri, Song, Wongpanich, Lewandowski, Abdelfattah; Google, 2025) — arXiv 2509.26476 [HTML]
*In-scope:* central claim is architectural — a 300M-parameter regression language model (RLM)
vs GNN-based graph-embedding baselines (Arch2Vec, CATE) and a prior SoTA (FLAN) on NAS-Bench
performance-ranking tasks (NASBench-101/201 and five NAS design spaces).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED — "No mention of multiple runs, random seeds, standard deviations, or confidence intervals for headline comparisons." |
| 2 Param-matched | **F** | RLM's own size is disclosed — "a relatively small 300M parameter RLM initialized from T5Gemma" — but baseline sizes are NOT STATED for FLAN, Arch2Vec, or CATE. GNN-based graph-embedding baselines (Arch2Vec/CATE) are conventionally orders of magnitude smaller than a 300M-parameter LM; with no explicit capacity control or discussion, this is the classic "proposed model larger with no control" F pattern, just with only one side of the comparison disclosed. |
| 3 Significance test | **F** | NOT STATED — "No formal significance testing reported." |
| 4 Eval-set integrity | **∂** → **F** (human verification, §D) | Partial, uneven disclosure: some benchmarks state eval size ("We test 1024 programs per language" for CodeNet) but "specific dataset splits for train/val/test not uniformly reported across all benchmarks." |
| 5 Effect vs noise | **F** | The paper's strongest same-class comparison — against FLAN, the actual prior SoTA baseline — shows a negligible gap: "Table 4 shows average Kendall-τ: RLM = 0.461 vs FLAN = 0.459" (Δτ≈0.002), well inside the rubric's "<0.02 τ" sub-noise anchor. The "substantially outperforms" language in the abstract refers to comparisons against older/weaker baselines (Arch2Vec, CATE), not this near-tied comparison against the strongest baseline. |
*Fails 4 of 5 axes (1, 2, 3, 5); notable as a case where the paper's headline "outperforms"
framing is technically accurate against weak baselines but the effect size against the
actual strongest same-class competitor is within noise.*

### A17. Omniwise (Wang, Ramos, Awad, Lowery; UIUC/AMD, 2025) — arXiv 2506.20886 [HTML]
*In-scope:* central architectural comparison is an internal backbone ablation — LLaMA 3.2 3B
vs StarCoder2 3B as the base model for an LLM-based GPU-kernel performance-counter predictor
(no external analytical/prior-ML baseline is compared against).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED — "Table 4 reports point percentages only (e.g., '89.9%') without error bars or uncertainty quantification." |
| 2 Param-matched | **∂** | Both backbones are the same nominal size class — "LLaMA 3.2: '3 billion parameters'... StarCoder-2: '3B' (implied 3 billion)" — a rough, stated parity similar to the A9 GraphSAGE-vs-GAT precedent, but not an exact/verified match nor a stated training-budget match. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | *Corrected evidence:* Table 4 contains only Omniwise's own accuracy within error thresholds; there are NO StarCoder2 numbers anywhere. The LLaMA-3.2-3B-over-StarCoder2 choice is asserted in §2.2.3 after unspecified "studies" — less than point estimates. Training: 1 epoch, seed 42 (Table 3). |
| 4 Eval-set integrity | **P** | "The dataset was split into 90% for training and 10% for validation, with around 4000 kernels reserved for testing" — an explicit, non-single-digit held-out test set, clearly separated from training/validation. |
| 5 Effect vs noise | **F** | The comparative claim itself is unquantified — "LLaMA 3.2 3B has exceeded StarCoder2 3B in the accuracy of predictions" is asserted with no specific delta number surfaced in the fetched text, and no external (non-LLM) baseline is compared at all. An unquantified "ours is better" claim with no magnitude given cannot be assessed against noise, so conservative scoring treats it as failing rather than UNCLEAR. |
*Fails 3 of 5 axes (1, 3, 5); notable for having NO external baseline at all — the entire
"architectural claim" is an unquantified internal backbone choice.*

### A18. HLS Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing (Wu, Yang, Xie, Li, Hao; DAC 2022) — arXiv 2201.06848 [PDF via ar5iv]
*In-scope:* central claim is architectural — benchmarks 14 different GNN architectures
(GCN/GIN variants, GAT, GGNN, RGCN, Graph U-Net, GNN-FiLM, etc.) against each other and against
prior work (Vitis HLS estimates, IronMan) for FPGA high-level-synthesis performance prediction.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **∂** | "Each model is trained with five runs using different random number seeds and we report the average of three with least validation error" — genuinely multiple seeds (5), but the reporting method is a soft best-of-N (dropping the 2 worst of 5 runs) rather than a full mean+dispersion, and "no standard deviations or confidence intervals reported for headline comparisons." Matches the rubric's ∂ "best of N only" anchor closely. |
| 2 Param-matched | **F** | All 14 GNNs share the same width/depth setting ("empirically set as five layers with a hidden-dimension size of 300"), a partial architectural standardization, but "the paper provides no model parameter counts or capacity-matching discussion" — different message-passing mechanisms (e.g. GAT's attention weights, RGCN's per-relation matrices) yield different true parameter counts at the same width, so shared hidden-dim is not a verified capacity match; and the headline external comparisons (vs Vitis HLS, IronMan) have no capacity control discussed at all. Conservative scoring keeps this at F given the ambiguity. |
| 3 Significance test | **F** | NOT STATED — "No significance tests, p-values, or statistical comparisons provided." |
| 4 Eval-set integrity | **P** | "DFG and CDFG datasets are randomly split into 80% train, 10% validation and 10% test," over a synthetic corpus of "19,120 and 18,570 C programs" respectively, plus named real-world benchmark suites ("MachSuite 16, CHStone 10, PolyBench 30" = 56 real applications). Clear, adequately-sized held-out split. |
| 5 Effect vs noise | **∂** → **F** (human verification, §D) | Large claimed gains — "outperforms HLS by up to 40× on resource prediction" and "excels existing predictor IronMan by 2× to 5×" — against baselines that are legitimate prior tools/work (not obvious strawmen), but with no variance/significance reporting for either the internal 14-GNN comparison or the external Vitis HLS/IronMan comparison, conservative scoring keeps this at partial. |
*[Corrected 2026-10-11: after the §D human check moved Axis 5 ∂ → F, A18 fails **3** of 5 (2, 3, 5); §D's totals already count it this way, so 17/20 is unchanged. The text below predates that correction.] Originally: fails 2 of 5 axes (2, 3) — tied with A9 for the least-bad rigor profile in the corpus so far,
largely on the strength of a genuine multi-seed (if not fully dispersion-reported) protocol and
a clean, sizeable held-out split.*

### A19. Toward Compiler World Models: Learning Latent Dynamics for Efficient Tensor Program Search (Pan, Huang, Zhou, Xing, Wu; 2026) — arXiv 2606.09312 [HTML]
*In-scope:* central claim is architectural — a learned "action-conditioned latent dynamics"
world model for guiding tensor-program search vs Ansor's default (XGBoost-based) cost-model
search, evaluated as an end-to-end tuning-time/quality comparison.

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED — "The paper reports no standard deviations, confidence intervals, or multiple runs. Single measurements are presented as point estimates across benchmarks." |
| 2 Param-matched | **F** | NOT STATED — "No model size comparisons or parameter counts provided for any learned components (encoder, transition model, ranking model) versus baselines." |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — "No statistical tests, p-values, or significance measures reported." |
| 4 Eval-set integrity | **∂** | Training/validation data sizes are given in detail (e.g. "391,240 / 395,761 state-action samples, yielding 312,992 / 316,608 training samples and 78,248 / 79,153 validation samples") but "no test set reported; evaluation uses 22 representative subgraphs from 103 total extracted subgraphs" — an eval set with no clean train/val/test separation described, and a fairly small (22-subgraph) headline evaluation pool with no power caveat. |
| 5 Effect vs noise | **∂** | Reasonably large, plausible gains against a real prior-SOTA baseline (Ansor) — "1.37× speedup on GPU and 1.54× on CPU under the same 64-trial budget," and "Ansor-64 baseline: '3.91 ms' vs. their method: '3.02 ms'" — but reported "without error bars," so the gap cannot be formally checked against noise. |
*Fails 3 of 5 axes (1, 2, 3); this is also the only paper this round making its architectural
comparison against Ansor's cost model directly (rather than against a GNN/analytical baseline),
extending the corpus's coverage of the TenSet/TLP/Ansor tuning-search lineage.*

### A20. PerfVec (Li et al.; SC 2024) — arXiv 2310.16792 [HTML]
*In-scope:* central claim is an **internal architecture ablation** — comparing neural encoder
choices (linear regression, MLP, GRU, biLSTM, Transformer, vs the chosen 2-layer 256-dim LSTM)
for learning generalizable microarchitecture/program representations; notably, the paper
explicitly declines to make a rigorous head-to-head claim against prior SOTA (Ithemal, GRANITE).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **F** | NOT STATED for the ablation. "Results appear to be single runs without variance reporting." (Figure 3's "standard deviations across microarchitectures" is cross-microarchitecture spread, not seed/run variance — does not count.) |
| 2 Param-matched | **∂** | The paper runs an explicit **capacity sweep**: "The leftmost portion of Figure 6 explores neural architectures other than the default unidirectional LSTM, including linear regression, multilayer perceptron (MLP), gated recurrent unit (GRU), bidirectional LSTM (biLSTM), and Transformer encoder," plus a depth/width sweep concluding "using more than two layers or higher than 256-dimensional representations brings negligible or no accuracy improvement" — this is genuinely closer to the rubric's P anchor ("a capacity sweep showing the gain is not just scale") than most of the corpus, though exact parameter counts across the different architecture *types* (e.g. Transformer vs LSTM at the same width) are not equalized or reported, so it stops short of a verified match. |
| 3 Significance test | **F** (human-confirmed 2026-10-05) | NOT STATED — no significance testing for the ablation comparisons. |
| 4 Eval-set integrity | **P** | "roughly 90% of them are dedicated for training, 5% for validation, and 5% for testing" from a "737 million instructions" / "2.0 TB" dataset — clear split, very large corpus, though the exact test-set instance count is not spelled out. |
| 5 Effect vs noise | **∂** | Ablation conclusions are qualitative rather than numeric in the fetched text ("linear model incurs the largest error," "Transformer has the second largest error, and larger Transformer models fail to converge in training") with no reported variance to check against noise. Notably, on the question of prior-SOTA comparison, the paper is unusually candid: it states outright that "different hardware platforms were used in these papers so this is not a rigorous comparison" when discussing Ithemal/GRANITE — i.e. it explicitly refuses to claim a controlled win over the established baselines, rather than asserting one anyway. |
*Fails 2 of 5 axes (1, 3); the second-least-bad rigor profile in the corpus alongside A9/A18,
and the strongest candor counter-example found this round — it explicitly flags its own
SOTA comparison as non-rigorous rather than asserting a win.*

## §B — Emerging signal (HYPOTHESIS ONLY, n=2 — NOT a result)
Both TPUGraphs-lineage papers **fail Axes 2 (param-match) and 3 (significance)** outright;
neither attributes its architectural comparison to the idea vs capacity. This is *consistent
with* the project thesis — but **n=2 proves nothing**; recorded only to show the audit is
already finding the predicted pattern. Do not quote any rate until §C is done.

## §C — Audit queue (to score; do NOT pre-fill from memory)
Real arXiv IDs where confirmed; `[HTML]` = WebFetch-able now, `[PDF]` = needs human read.
Scored round 1 (moved to §A3–A9): ~~GRANITE (2210.03894)~~, ~~Ithemal (1808.07412)~~,
~~ProGraML (2003.10536)~~, ~~TLP (2211.03578)~~, ~~PerfSAGE (2301.10999)~~,
~~NeuSight (2407.13853)~~, ~~Kaufman et al. TPU learned perf model (2008.01040)~~.
Scored round 2 (moved to §A11–A20): ~~TenSet (no arXiv, NeurIPS D&B PDF)~~,
~~Polynormer (2403.01232)~~, ~~Reconfigurable-dataflow cost model (2511.01872)~~,
~~Facile (2310.13212)~~, ~~COGNATE (2506.00424)~~, ~~RLM-for-Code (2509.26476)~~,
~~Omniwise (2506.20886)~~, ~~HLS-GNN benchmarking (2201.06848)~~,
~~Toward Compiler World Models (2606.09312)~~, ~~PerfVec (2310.16792)~~.

**Round-2 resolutions of round-1 stuck items:**
- **TenSet** — RESOLVED (scored as A11). No arXiv ID exists (confirmed again this round); the
  official NeurIPS D&B proceedings PDF (datasets-benchmarks-proceedings.neurips.cc) was
  successfully read via the r.jina.ai text-extraction proxy after direct WebFetch of both the
  PDF and the OpenReview page failed (binary garble / CAPTCHA wall respectively).
- **Polynormer** — RESOLVED (scored as A12). Appendix L was unreachable via plain WebFetch of
  either the ar5iv HTML or the raw PDF (truncated before the appendix on 3 separate attempts
  this round, 2 last round); the r.jina.ai proxy fetch of the raw PDF finally surfaced it.
- **Reconfigurable-dataflow paper (2511.01872)** — RESOLVED (scored as A13). Confirmed the
  round-1 "suspiciously thin" extraction was a genuine WebFetch/PDF-parsing artifact, not the
  paper's actual content — the r.jina.ai proxy surfaced substantially more text. Some fields
  (exact train/test split, parameter counts) are still genuinely NOT STATED in the paper itself.
- **Halide auto-scheduler (Adams et al. 2019)** — STILL STUCK / ruled effectively
  out-of-workflow. Confirmed again this round: no arXiv ID, PDF-only via
  halide-lang.org/papers/halide_autoscheduler_2019.pdf. Not attempted further per the task's
  guidance to deprioritize it; would need a manual (non-WebFetch) PDF read to ever score.

**New unscored candidates found this round (not yet fetched in enough depth to score):**
- AUTOCOMP (Hong, Bhatia, Cheung, Shao; MLSys) — arXiv 2505.18574 — likely **out of scope**:
  an LLM-driven in-context search optimizer with "no model training," not a learned-cost-model
  architecture comparison; would need a closer read to confirm before excluding for good.
- DiffHLS — arXiv 2604.09240 — HLS QoR prediction with GNNs + LLM code embeddings, found via
  search snippet only, not yet fetched.
- Intelligent4DSE — arXiv 2504.19649 — HLS design-space exploration with GNNs + LLMs, found
  via search snippet only, not yet fetched.
- TCL (continual learning for cross-hardware tensor program optimization) — arXiv 2604.12891 —
  found via search snippet only, not yet fetched.
- GPU Forecasters (LMs as selective surrogates for kernel runtime) — arXiv 2605.31464 — found
  via search snippet only, not yet fetched.
- "Can Large Language Models Predict Parallel Code Performance?" — arXiv 2505.03988 — found
  via search snippet only, not yet fetched.
- + further Kaggle "predict-ai-model-runtime" competition writeups (2nd place, 14th place, etc)
  exist on kaggle.com/competitions/predict-ai-model-runtime/writeups — not yet assessed for
  whether they're citable/scoreable as "papers" under this rubric (they're competition
  writeups, not peer-reviewed or arXiv'd — would need a scope decision before scoring).
*(Target corpus ≈30. Current: 20 scored. Each row needs: venue, year, task, 5 axis scores w/
quotes, and "our-conclusion-affected?".)*

## §D — What completing this requires (before any % is quoted)
1. Score ≥~20 papers from actual text (HTML pre-screen + **human read of the PDF-only ones**). ✅ done (20/20).
2. **Two independent raters** on a ≥20% sample; report Cohen's κ (rubric fairness control).
   **⚠️ PARTIALLY done, 2026-08-19 — pilot signal only, does NOT fully satisfy this item.**
   A fresh, context-isolated AI agent blind-scored the same 5-paper sample (ProGraML, TenSet,
   Facile, Regression LMs for Code, HLS-GNN Benchmarking — 25% of the corpus, randomly selected
   via `random.seed(20260819)`, see `phase7_second_rater_packet.md`) without seeing rater-1's
   scores. **Result: 21/25 = 84% raw agreement, Cohen's κ ≈ 0.67 pooled (substantial agreement,
   Landis & Koch scale).** Per-axis: Axis 1 (seeds) κ=1.0 (perfect — both raters found every
   paper fails); Axis 3 (significance) raw agreement 5/5 but κ undefined (both raters said F on
   every paper — zero marginal variance, a known kappa-paradox case, not a scoring failure);
   Axis 2 (param-match) 80% raw but κ=0 (same kappa-paradox: rater-1 said F on all 5, one
   disagreement on HLS-GNN pulls κ to exactly the chance level despite high raw agreement);
   Axis 5 (effect-vs-noise) κ≈0.55 (moderate, one disagreement on TenSet); **Axis 4 (eval-set
   integrity) is the weakest, κ≈−0.11** (two disagreements out of 5 — Facile ∂-vs-F and HLS-GNN
   P-vs-∂ — both cases where rater-2 scored stricter/more conservative than rater-1 on the same
   underlying evidence, suggesting Axis 4's anchors may need sharpening, per the rubric's own
   "if two raters disagree, sharpen the anchor" instruction).
   **✅ Human-verification pass on this same 5-paper sample completed 2026-08-25** (source-
   grounded: each paper's actual arXiv/proceedings text was re-fetched and the AI raters'
   quotes independently re-derived, not just read-and-agreed). Axes 1–3 were confirmed as
   originally scored for all 5 papers (unanimous F, with one RLM-for-Code Axis 4 correction
   ∂→F found in a prior 2026-08-24/25 pass — the paper admits explicit train/test overlap on
   CodeNet). Axis 5 was the focus of this pass and resolved the pilot's two live disagreements
   and one Axis 5 blind spot:
   - **TenSet, Axis 5: ∂ → F.** Both raters had cited the same Table 4 top-k scores but reached
     different verdicts; a full per-network breakdown shows the "best" model (MLP+Ranking)
     actually *loses* to MLP+MSE on ResNet-50 (0.8823 vs 0.8873), and margins are ≤0.02 or
     negative on 3 of 5 networks. Zero variance reported anywhere. F is the better-supported
     read of the rubric's own "<0.02, unknown variance" anchor.
   - **HLS-GNN Benchmarking, Axis 5: ∂ → F.** Neither AI rater caught this: the paper's "2×–5×
     vs IronMan" headline claim is not an actual measured comparison — IronMan never appears as
     a row in the paper's own results table (Table 5); the claim is a transitive inference
     ("IronMan ≈ GCN, GCN < RGCN, therefore IronMan < RGCN by the same margin"). Only the Vitis
     HLS leg of the claim (confirmed, up to 40–51×) is a real measured result.
   - ProGraML and Facile: both raters' quoted *numbers* had minor inaccuracies (a chip-splicing
     error in Facile's Ithemal comparison, an unsourced "~1%" figure) but the final ∂ verdict on
     both survived correction — logged as quote-accuracy discrepancies, not score changes.
   **Net: 3 of 25 axis-scores checked (12%) were corrected on human verification, all toward
   stricter (never looser) scores** — no evidence the AI-only-scored remaining 15/20 papers are
   biased favorably, but also no direct check on them. **This satisfies the rubric's ≥20% sample
   requirement with full human verification (not just a second AI pass); the remaining 15/20
   papers (75% of the corpus) remain AI-prescreened only and have not been independently
   re-checked by a human or a second rater.** State this scope explicitly wherever §D's stats
   are cited in the paper.
   **✅ Axis 3 human check on the remaining 15 papers completed 2026-10-05** (the first author, against each
   paper's PDF, with a check sheet; results in each §A Axis 3 row). Axis 3 is now human-verified
   for **all 20** papers. One score changed: **A2 TGraph F → ∂** (Table 2 reports ±1-sigma for every
   cell of the headline; no test). This is the first correction toward a *looser* score; all
   earlier corrections went stricter. A10 AutoTVM shows unlabeled shaded bands in Figs. 4–5; kept
   F under the rubric's "when in doubt, weaker category" rule because the bands are never defined
   (A10's Axis 1 evidence corrected: not "single curves"). Evidence corrected for A17; claim-accuracy
   notes logged for A12 and A15 (Axis 5 material, scores unchanged). Axes 1, 2, 4, 5 remain
   AI-prescreened for those 15 papers.
3. **Headline stats (computed 2026-08-25; recomputed 2026-10-05 after the Axis 3 check).** n=20. Uses AI-prescreened (rater-1) scores for the
   15 papers outside the verification sample, and the corrected scores above for the 5 that were
   human-verified (TenSet Axis 5=F, HLS-GNN Axis 5=F, RLM-for-Code Axis 4=F; all other cells
   unchanged from §A).

   | Axis | F | ∂ | P | % F | % (∂ or F) |
   |---|---|---|---|---|---|
   | 1 Seeds/variance | 16 | 4 | 0 | **80%** | 100% |
   | 2 Param-matched | 17 | 3 | 0 | **85%** | 100% |
   | 3 Significance test | 19 | 1 | 0 | **95%** | 100% |
   | 4 Eval-set integrity | 4 | 7 | 9 | **20%** | 55% |
   | 5 Effect vs noise | 6 | 14 | 0 | **30%** | 100% |

   - **Compound failure:** 17/20 papers (**85%**) fail ≥3 of 5 axes outright (was 18/20 = 90%
     before A2's Axis 3 correction on 2026-10-05). Three papers fall below that bar: A9 (Kaufman
     et al., MLSys 2021 — fails only Axis 3), A20 (PerfVec, SC 2024 — fails Axes 1 and 3 only) and
     A2 (TGraph, 2024 — fails Axes 2 and 4 only).
   - **No paper runs a significance test:** 20/20 (Axis 3 is F for 19 and ∂ for 1; the ∂ reports
     variance but still no test). Quote this, not "100% fail Axis 3".
   - **Claim-level / §6 money quote:** because no paper in the corpus runs a significance
     test (Axis 3 is never P), **0/20 papers (0%) have Axes 1–3 all P** — not one of the 20 audited papers'
     headline architectural comparisons, as reported, would survive a parameter-matched,
     multi-seed, statistically-tested re-run. Draft §6 sentence: *"Across 20 recently published
     learned-cost-model / GNN-for-compilers papers making comparative architectural claims,
     100% report no statistical significance test for their headline comparison, 80–85% report
     no seed/variance data or capacity-matched baseline, and 85% fail at least 3 of 5 rigor axes
     outright — not one of the 20 papers' central architectural claims, as reported, would
     survive a parameter-matched, multi-seed, statistically-tested comparison."*
4. **Rigorous counter-examples** (for calibration, per the rubric's fairness-control
   requirement — include these alongside the headline % so §6 reads calibrated, not
   adversarial):
   - **A9** (Kaufman, Phothilimthana et al.; MLSys 2021) — the least-bad profile in the corpus:
     fails only Axis 3 outright; runs two independent held-out split protocols (random +
     hand-picked "hard" split); reports min/max dispersion for a secondary result; and is the
     *only* paper in the corpus that honestly reports its own method losing to the baseline on
     one task/split rather than only showing the favorable comparison.
   - **A20** (PerfVec; SC 2024) — fails only Axes 1 and 3; runs an explicit capacity sweep
     across encoder architectures (linear/MLP/GRU/biLSTM/Transformer), the closest thing in the
     corpus to Axis 2's "P" anchor; and explicitly declines to claim a rigorous SOTA comparison
     against Ithemal/GRANITE ("different hardware platforms were used in these papers so this is
     not a rigorous comparison") rather than asserting an unearned win.
   - **A18** (HLS-GNN Benchmarking; DAC 2022) — the one genuine multi-seed training protocol in
     the corpus (5 seeds, Axis 1 = ∂ not F), even though its external-baseline headline claim
     did not hold up under this session's verification (Axis 5 → F, above) — a useful reminder
     that rigor on one axis doesn't imply rigor on all of them.

---
*Provenance:* scorings A1–A2 from WebFetch of the arXiv HTML on 2026-06-29; quotes are the
fetch model's extraction and must be human-confirmed against the source before publication.
