# Phase 7 — Literature Meta-Analysis (evaluation rigor audit)

> **STATUS: STARTED 2026-06-29. 2 / ~30 papers VERIFIED. NO headline percentage yet —
> and none may be quoted until the audit is completed with human verification (see §5).**
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
| 3 Significance test | **F** | NOT STATED — no t-test/Wilcoxon/bootstrap. |
| 4 Eval-set integrity | **∂** | "We split the data using 80-10-10 ratio by graphs"; reports test τ (held-out test exists ✓) but few eval graphs per collection + "report final metrics only on 1,000 configurations", no power caveat. |
| 5 Effect vs noise | **F** | "GraphSAGE vs GCN … comparable" with no numbers/variance → gain-vs-noise not assessable. |
*Fails ≥3 of 5 axes (1,2,3,5).*

### A2. Cross-attention "TGraph" (tensor-compiler GNN, 2024) — arXiv 2405.16623 [HTML]
*In-scope:* central claim is architectural (configuration cross-attention beats baselines).

| Axis | Score | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | **∂** | Reports dispersion ("confidence ranges … 1-sigma", "0.6840 ± 0.0110") — better than most — but via "K-fold cross-validation with K=20" and "pick the top-4 folds by the validation score" (fold selection, not independent seeds). |
| 2 Param-matched | **F** | NOT STATED — "does not report model sizes, parameter counts, or capacity comparisons". |
| 3 Significance test | **F** | NOT STATED — no hypothesis test for the improvement. |
| 4 Eval-set integrity | **F** | "TpuGraphs … does not provide public test data annotations. Hence, we report the cross-validation score" on "merged train/validation" AND "pick the top-4 folds by the validation score" → selection leakage, no held-out test. |
| 5 Effect vs noise | **∂** | TGraph τ=0.674 vs Xu-2023 0.610 (Δ≈0.064) with ±~0.011 ranges → gain exceeds its own σ, but no param-match so not attributable to the idea; baseline reasonable. |
*Fails 2–3 of 5; notably has dispersion (Axis 1 ∂) but a leaky eval (Axis 4 F).*

## §B — Emerging signal (HYPOTHESIS ONLY, n=2 — NOT a result)
Both TPUGraphs-lineage papers **fail Axes 2 (param-match) and 3 (significance)** outright;
neither attributes its architectural comparison to the idea vs capacity. This is *consistent
with* the project thesis — but **n=2 proves nothing**; recorded only to show the audit is
already finding the predicted pattern. Do not quote any rate until §C is done.

## §C — Audit queue (to score; do NOT pre-fill from memory)
Real arXiv IDs where confirmed; `[HTML]` = WebFetch-able now, `[PDF]` = needs human read.
- GRANITE — basic-block throughput GNN — arXiv **2210.03894** `[PDF / human-read]`
- Ithemal — basic-block throughput LSTM — arXiv **1808.07412** `[PDF / human-read]`
- PROGRAML — program graph representations — *(locate arXiv id)* `[?]`
- Ansor / AutoTVM learned cost models — *(locate ids)* `[?]`
- Halide auto-scheduler (Adams 2019) — *(locate)* `[?]`
- TenSet cost-model benchmark — *(locate)* `[?]`
- + remaining recent TPUGraphs / layout-cost-model follow-ups (Kaggle-comp solutions, 2024–25)
  — prioritize `[HTML]` arXiv versions for machine pre-screen.
*(Target corpus ≈30. Each row needs: venue, year, task, 5 axis scores w/ quotes, and
"our-conclusion-affected?".)*

## §D — What completing this requires (before any % is quoted)
1. Score ≥~20 papers from actual text (HTML pre-screen + **human read of the PDF-only ones**).
2. **Two independent raters** on a ≥20% sample; report Cohen's κ (rubric fairness control).
3. Then compute the headline stats: % F per axis, % failing ≥3 axes, and the fraction whose
   central architectural claim rests on an uncontrolled comparison (Axes 1–3 not all P).
4. Include rigorous counter-examples (papers that DO param-match / multi-seed) so §6 reads
   calibrated, not adversarial.

---
*Provenance:* scorings A1–A2 from WebFetch of the arXiv HTML on 2026-06-29; quotes are the
fetch model's extraction and must be human-confirmed against the source before publication.
