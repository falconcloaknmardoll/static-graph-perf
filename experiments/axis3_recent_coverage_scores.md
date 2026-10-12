# Axis 3 check on recent work (Frame B): scoring sheet

> Scorer: **the first author (human), from the PDF.** Rules: `axis3_recent_coverage_declaration.md` (frozen
> e12c51a). This sheet was committed **blank**.
>
> **How to record:**
> - Score in list order, B1 → B10.
> - Fill each block at the time of reading and commit after each paper (or each sitting), so git
>   shows the order and timing.
> - Do not edit a block after its commit. A correction is a dated erratum appended under the
>   block, keeping the original.
> - Claude does not pre-fill, pre-screen or suggest anything here (declaration §4).

## Quick reference (from the declaration; the declaration governs if they differ)

**Step 1 — Find the qualifying comparison (§3).** Fix it *before* looking at results.
- It must compare the paper's cost or performance model against a baseline model where **only
  the model differs**. The metric may be accuracy, ranking, or tuning outcome.
- These **do not qualify** (whole-system):
  - comparisons against frameworks, vendor libraries or compilers;
  - any comparison that changes **anything besides the model** (search, search space, budget …);
  - a comparison where the paper is silent on whether anything else differs.
- **Which one is the headline:** the first qualifying comparison in the abstract; else the first
  in the Introduction's contributions; else the first in the evaluation section. If its numbers
  appear in several tables, score the table the abstract or Introduction cites.
- **No qualifying comparison anywhere → N/A** (record what each baseline changes).

**Step 2 — Score Axis 3 on that comparison** (`meta_analysis_rubric.md:48-54`).
- **P:** a named test, or a statistical/bootstrap CI on the comparison, with a reported statistic
  or p-value, on the **headline** comparison.
- **∂:** variance, std, ± or defined error bars on the headline, but no test.
- **F:** no variance and no test. This includes "significantly" with no statistic, undefined
  bands, and a test on a secondary comparison only.
- **When in doubt, score the weaker category.**
- **Where to look:** main PDF plus any appendix inside it. Do not search external supplements.
- **Version:** the version of record if obtainable, else the latest arXiv version.

**Other outcomes:** *not obtained* (no PDF by Oct 21) · *not scored (time)*. Neither is ever
replaced.

---

## B1. LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers
*2024, PACT · arXiv 2403.11522*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B2. Learning to Compare Hardware Designs for High-Level Synthesis
*2024, MLCAD · arXiv 2409.13138*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B3. Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning
*2024, ASPLOS · arXiv 2402.02361*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B4. Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment
*2024, ICS · doi 10.1145/3650200.3656631*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B5. Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs
*2024, DAC · doi 10.1145/3649329.3658249*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B6. Fast and Accurate Context-Aware Basic Block Timing Prediction using Transformers
*2024, CC · doi 10.1145/3640537.3641572*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B7. NLTSP: A cost model for tensor program tuning using nested loop trees
*2025, J. Syst. Archit. · doi 10.1016/j.sysarc.2024.103305*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B8. UniCoMo: A Unified Learning-Based Cost Model for Tensorized Program Tuning
*2024, ICCD · doi 10.1109/ICCD63220.2024.00081*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B9. UniCoX: A Unified Cost Model for Tensorized Program Tuning Across Ubiquitous Accelerators
*2026, IEEE TC · doi 10.1109/TC.2025.3626148*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## B10. UniSparTa: A Unified Sparse Tensor Program Tuning Framework
*2026, IEEE TCAD · doi 10.1109/TCAD.2025.3636442*

- **Version read:** (version of record / arXiv vN)
- **Date read:**
- **Minutes spent:**
- **Headline comparison (verbatim + page + where: abstract / intro / eval).** Fill this before
  reading the results:
  > 
- **Qualifies? (only the model differs)** yes / no. Quote or table showing what the baseline
  changes:
  > 
- **Axis 3 score:** P / ∂ / F / N/A / not obtained / not scored (time)
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > 
- **Notes** (tie-break applied? anything ambiguous?):

---

## Summary (fill after B10, or at the Oct 23 cutoff)

| # | Score | Headline comparison (short) | Evidence location |
|---|---|---|---|
| B1 |  |  |  |
| B2 |  |  |  |
| B3 |  |  |  |
| B4 |  |  |  |
| B5 |  |  |  |
| B6 |  |  |  |
| B7 |  |  |  |
| B8 |  |  |  |
| B9 |  |  |  |
| B10 |  |  |  |

- **Counts:** P __ · ∂ __ · F __ · N/A __ · not obtained __ · not scored __
- **n scored** (excluding N/A, not obtained and not scored) = __
- **Runs a significance test (P):** k = __ of n = __
- **§8 sentence:** the matching option from declaration §6, chosen by counts only.
- **Appendix must also state:** the B8–B10 clustering, the N/A count, "most-cited *in this
  frame*" (2–16 citations), and the §7 blindness statement.
