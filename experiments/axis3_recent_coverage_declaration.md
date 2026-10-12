# Axis 3 check on recent work (Frame B): declaration

> **STATUS: FROZEN 2026-10-12** (approved by the first author; drafted by Claude). Binding from the commit
> that adds this file. No paper on the list below was opened (full text) before that commit. All
> 10 PDFs were downloaded, unopened, before freezing. Any change after freezing is a dated
> amendment appended at the end, made before the affected paper is read; the text above it is
> not edited. Rationale and options: `audit_coverage_assessment_2026-10-12.md`.

## 1. Question

For each of the 10 most-cited in-scope 2024–2026 papers in the registered coverage frame
(Frame B, `phase7_coverage/screening.md:84-95`): **does the paper run a statistical
significance test for its headline comparative claim?** This is Axis 3 only. No other axis is
scored.

## 2. Frozen paper list (scored in this order)

| # | S2 rank | Title | Year | Venue (S2) | ID |
|---|---|---|---|---|---|
| B1 | 319 | LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers | 2024 | PACT | arXiv 2403.11522 |
| B2 | 384 | Learning to Compare Hardware Designs for High-Level Synthesis | 2024 | MLCAD | arXiv 2409.13138 |
| B3 | 387 | Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning | 2024 | ASPLOS | arXiv 2402.02361 |
| B4 | 423 | Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment | 2024 | ICS | doi 10.1145/3650200.3656631 |
| B5 | 433 | Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs | 2024 | DAC | doi 10.1145/3649329.3658249 |
| B6 | 439 | Fast and Accurate Context-Aware Basic Block Timing Prediction using Transformers | 2024 | CC | doi 10.1145/3640537.3641572 |
| B7 | 443 | NLTSP: A cost model for tensor program tuning using nested loop trees | 2025 | J. Syst. Archit. | doi 10.1016/j.sysarc.2024.103305 |
| B8 | 444 | UniCoMo: A Unified Learning-Based Cost Model for Tensorized Program Tuning | 2024 | ICCD | doi 10.1109/ICCD63220.2024.00081 |
| B9 | 451 | UniCoX: A Unified Cost Model for Tensorized Program Tuning Across Ubiquitous Accelerators | 2026 | IEEE TC | doi 10.1109/TC.2025.3626148 |
| B10 | 452 | UniSparTa: A Unified Sparse Tensor Program Tuning Framework | 2026 | IEEE TCAD | doi 10.1109/TCAD.2025.3636442 |

**Rules for the list:**
- **No substitutions.** The list is exactly Frame B as registered. No paper is added, swapped,
  or dropped for any reason.
  - Intelligent4DSE (S2 #467) ties B9 and B10 at 2 citations. The registered tie-break (S2 file
    order, `phase7_coverage/screening.md:31-33`) puts it outside. It stays outside.
- **Known clustering.** B8, B9 and B10 share 6–8 authors (`candidates.json`). B9 may be the
  journal extension of B8. They are scored separately as listed. The appendix states the
  clustering.

## 3. What is scored

**Headline comparative claim.** Fixed by rule before reading. The rule is copied from how the
20-paper corpus was scoped (precedents cited below), not written for these 10.

- **What qualifies:** a comparison of the paper's cost or performance model against a baseline
  model, **where only the model differs**. The metric may be prediction accuracy, ranking quality,
  or tuning outcome.
  - *Precedent A6 (TLP)*, `phase7_meta_analysis.md:200-202`: in scope as a learned cost model
    vs Ansor's XGBoost cost model, measured on "tuning search efficiency". This is a tuning-outcome
    metric with only the model changed.
  - *Precedent A19 (Compiler World Models)*, `phase7_meta_analysis.md:417-419`: in scope as a
    learned model guiding search vs Ansor's cost-model search, "evaluated as an end-to-end
    tuning-time/quality comparison".
- **What does not qualify (whole-system comparisons):** comparisons against other frameworks,
  vendor libraries or compilers.
  - *Precedent A10 (AutoTVM)*, `phase7_meta_analysis.md:268-271`: scored only on its cost-model
    comparisons; "end-to-end systems claims against cuDNN/TensorFlow … are out of the rubric's
    scope".
- **Model plus something else:** if a comparison changes **anything besides the model** (for
  example the cost model *and* the search strategy, the search space, or the measurement budget,
  together), it counts as whole-system and does not qualify.
  - *Precedent Ansor*, `phase7_meta_analysis.md:84`: ruled out because its claim concerns the
    search space and search, not a cost model.
  - The scorer applies this from the paper's own description of what each baseline row changes.
    If the paper does not say whether anything besides the model differs, the comparison does
    **not** qualify (conservative: no inferring single-variable control from silence).
- **Which qualifying comparison is the headline:**
  1. the first qualifying comparison stated in the abstract;
  2. if the abstract has none, the first in the Introduction's contribution list;
  3. if neither has one, the first in the evaluation section.

  It is quoted verbatim, with page, in the scoring sheet *before* the paper's results are
  examined. If 1 or 2 names a qualifying comparison whose numbers appear in several tables, the
  table the abstract or Introduction cites is the one scored.

**A19 checked against this rule (2026-10-12): it qualifies, so the rule matches precedent.**
A19's text (arXiv 2606.09312 HTML, fetched 2026-10-12) states:
- that its model is "a drop-in scoring component inside the existing search loop" (Sec. 3.6),
  "replacing the default candidate-scoring pathway";
- "our method improves candidate evaluation rather than candidate generation or search-space
  design" (Sec. 6);
- that the comparison runs "under the same 64-trial budget" (Abstract; Sec. 4.6).

Its latent rollouts (Sec. 3.4) score candidates; the text describes no planning or lookahead
search built on them. So the paper itself says only the evaluator changed. A statement of this
kind from the paper is what satisfies "only the model differs".

*Caveat:* the paper does not define whether one "trial" is one hardware measurement in both
systems, and does not restate Ansor's search space. The rule's "silence" clause concerns whether
anything *besides the model* differs. A19 answers that explicitly, so the clause does not
apply.

*Quotes verified 2026-10-12* by exact-string search of the full arXiv HTML
(https://arxiv.org/html/2606.09312, downloaded 2026-10-12), not by a summarising fetch. All three
appear verbatim in the sections cited. The same full text gives further support:
- Sec. 3.6: the evaluator is connected "with the candidate generation and measurement stages
  already provided by the tensor-program search framework".
- Sec. 2.1.1: "Our work focuses on the candidate-evaluation stage".
- Sec. 4.6: the matched budget means "the numbers isolate the effect of changing the evaluator
  rather than the effect of using more measurements".

**Axis 3 anchors**, verbatim from `meta_analysis_rubric.md:48-54`:
- **Question:** "Is the claimed improvement supported by a hypothesis test, or asserted from a
  raw number being larger?"
- **P:** "appropriate test (paired t / Wilcoxon / bootstrap CI) with reported statistic,
  matched-seed pairing where applicable."
- **∂:** "reports variance but no test, leaving 'is the gap > noise?' to the reader's eye."
- **F:** "'ours = 0.36 vs theirs = 0.34, therefore better' with no variance or test."

**Operational clarifications.** These apply the existing anchors and precedents; they are not
new criteria.
- **P requires all three:**
  1. a named test, or a statistical or bootstrap confidence interval on the comparison;
  2. a reported statistic or p-value;
  3. that it applies to the **headline** comparison. A test on a secondary comparison only is
     not P for the headline.
- **The word "significant(ly)" without a statistic does not count toward P.** It applies the
  existing F anchor ("'ours = 0.36 vs theirs = 0.34, therefore better' with no variance or test",
  `meta_analysis_rubric.md:54`): a verbal claim of significance with no statistic is that case.
  **Confirmed by the first author 2026-10-12: this applies the existing anchor and is consistent with how the
  20 were scored. It is not a new rule, so §3a stays empty for it.**
- **Error bars, std or ± on the headline comparison, with no test, score ∂.** Precedent: A2,
  `phase7_meta_analysis.md:550-552`.
- **Shaded bands or error bars that are never defined score F.** Precedent: A10,
  `phase7_meta_analysis.md:553-555`.
- **Where to look:** the main PDF, including any appendix in the same document. External
  supplementary material is **not** searched. This matches how the 20 were scored (confirmed by
  The first author 2026-10-12).
- **Which version:** the version of record if obtainable, otherwise the latest arXiv version.
  Record which.

**Tie-break.** When in doubt, score the weaker category (`meta_analysis_rubric.md:12`;
`phase7_meta_analysis.md:23`).

**N/A rule (scope).** A paper is N/A if its full text contains **no qualifying comparison** as
defined above. This applies the rubric's own scope rule: papers making no comparative
architectural claim are recorded as N/A (`meta_analysis_rubric.md:17-19`).
- A paper whose only comparisons are whole-system (tuning speedup against other frameworks, or
  model and search changed together) is N/A. Screening flagged some papers as ambiguous;
  scoring does not pull them into scope.
- The N/A decision is recorded with the quote or table that shows what each baseline changes.
- N/A is reported, never replaced.

**Not obtained.** If a PDF cannot be obtained by **2026-10-21**, it is recorded as "not
obtained". It is reported, never replaced, and never scored from the abstract.

## 3a. Dated clarifications (rules not taken from existing precedent)

Anything the first author marks above as new rather than precedent is listed here before freezing, with the
date. Empty means every rule in §3 applies existing anchors or precedents.

- *(none yet)*

## 4. Who scores and how

- **Scorer:** the first author (human), from the PDF. Claude does not pre-screen these papers, extract
  quotes or suggest scores. Claude may prepare a blank scoring sheet only.
- **One sheet row per paper, recorded at the time of reading:**
  - the headline claim (verbatim + page);
  - the score (P / ∂ / F / N/A / not obtained);
  - the evidence quote, or "NOT STATED";
  - the version read;
  - the date;
  - the minutes spent.
- **Scores are final once entered.** A later correction is allowed only as a logged erratum
  that keeps the original.
- **No second rater.** The rubric does not require one for a supplement; the first author is the only
  scorer.
- **Timeline:**
  - freeze by 2026-10-17;
  - PDFs obtained, unopened, by 2026-10-18;
  - scoring 2026-10-19 to 2026-10-22, in list order;
  - results committed by **2026-10-23** (evidence cutoff).
  - If time runs out, the remaining papers (the lowest-ranked) are reported as "not scored
    (time)".

## 5. Reporting commitment

- **Every paper's outcome is reported in an appendix table, whatever it is:** P, ∂, F, N/A, not
  obtained, or not scored. The table gives title, venue, year and the evidence quote.
- **Never merged.** These results never enter the 20-paper headline numbers, tables or
  percentages, and are never combined into a "30 papers" figure.
- **Prose:** failures appear as counts only, and no paper is named as a failure
  (`phase7_meta_analysis.md:24-34`). A paper scoring P may be named as a counter-example, as
  A9, A20 and A18 are (`phase7_meta_analysis.md:586-602`).

## 6. How results enter §8 (one sentence + appendix table)

The sentence is chosen by outcome, not written after the fact. n = papers scored (excluding N/A,
not obtained and not scored).

- **If 0 of n:** "In a separate, pre-declared check of the 10 most-cited in-scope papers from
  2024–2026 in the same frame, none of the n we could score runs a significance test for its
  headline comparison (Appendix X)."
- **If k ≥ 1 of n:** "In a separate, pre-declared check of the 10 most-cited in-scope papers from
  2024–2026 in the same frame, k of the n we could score run a significance test for their
  headline comparison (Appendix X)." The P papers are added to the §8 counter-example
  paragraph.
- **If n ≤ 5:** the same sentence with the counts, followed by: "too few to support a
  conclusion about recent work."
- **In every case:**
  - "most-cited" always means most-cited *in this frame*. The appendix states the range: 2–16
    citations at retrieval (`phase7_coverage/screening.md:45`). The wording must not imply these
    are the field's landmark papers.
  - The appendix notes the B8–B10 clustering, the number of N/A papers and the blindness
    statement (§7).

## 7. Blindness statement (goes in the appendix verbatim)

> The scorer had not read the full text of any of the 10 papers before this declaration was
> frozen. During coverage screening (2026-10-05) the scorer saw each paper's title and, where
> Semantic Scholar had one, its abstract. None of the abstracts reports seeds, error bars,
> dispersion, confidence intervals or a hypothesis test. Several use "significantly" or
> "significant improvements" in a non-statistical sense, and these words were seen. One paper
> (NLTSP) had no abstract in Semantic Scholar. The scorer may know some of these papers from
> general reading; that familiarity cannot be ruled out, only the absence of a full-text read
> before freezing. Axis 3 was scored under anchors and tie-break rules fixed before the
> 20-paper audit and unchanged since.

*§3a is empty (2026-10-12): the "significantly" rule and the supplement scope were both confirmed
as existing practice, and the scope rule matches A6, A10, A19 and Ansor. The last sentence stands
as written.*

## 8. Approval

- [x] A19 checked against the "model plus something else" rule (§3): qualifies; no §3a entry
      needed (2026-10-12, AI fetch).
- [x] A19 quotes verified verbatim against the full arXiv HTML (2026-10-12, exact-string search).
- [x] "Significantly" rule: application of the existing anchor (the first author, 2026-10-12).
- [x] Supplement scope: main PDF + in-PDF appendix only, as for the 20 (the first author, 2026-10-12).
- [x] All 10 PDFs downloaded, unopened, all obtained (the first author, 2026-10-12).
- [x] the first author approves the list, the rules and the wording — date: 2026-10-12.
- [x] Renamed to `axis3_recent_coverage_declaration.md` and committed before any listed PDF was
      opened. The freeze commit is the first commit that adds this file: `git log --diff-filter=A
      -- experiments/axis3_recent_coverage_declaration.md`. Its hash is also recorded in
      `DEFENSIBILITY.md` (O15 log).
- [x] Log line added to the O15 entry in `DEFENSIBILITY.md`.
