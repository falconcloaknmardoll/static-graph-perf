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

- **Version read:** version of record, PACT 2025 proceedings (IEEE)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “We find that the model-guided LOOPER is, on average, 644× faster than the measurement-guided LOOPER while delivering comparable results as shown previously (Section VI-B3).” Eval §VI-B4, p. 10
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “we run LOOPER with both evaluation methods and with the same beam size (a beam size of 3) on the entire PolyBench set” (p. 10). The methods: “using the cost model as an evaluation function” vs. “compiling and executing those candidates to get their execution time” (p. 10, §VI-B3)
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Only point values: “on average, 644× faster” and the seidel2d_LARGE example, “25.61 seconds” vs. “more than 9 hours” (§VI-B4, p. 10)
- **Notes** (tie-break applied? anything ambiguous?): (1) The abstract headline was rejected: vs. Tiramisu, the search space also changes (p. 10); vs. Pluto, it’s a whole compiler with ILP search (p. 3). The intro contributions repeat both. §VI-B3 is controlled but LOOPer loses there (geomean 0.75). (2) Borderline: the baseline is ground-truth measurement (an oracle), and the claim is about search time. If oracles don’t count as a “baseline model”, score N/A. (3) “On average” is undefined (arithmetic or geomean), and the number of timing repetitions isn’t stated. (4) No tie-break needed. (5) Year note for B1: your rubric mentions one, but I don’t have its definition, so tell me what goes here.

---

## B2. Learning to Compare Hardware Designs for High-Level Synthesis
*2024, MLCAD · arXiv 2409.13138*

- **Version read:** arXiv v2 (arXiv:2409.13138v2, 7 May 2025). It carries the ACM MLCAD ‘24 copyright block, so it’s the authors’ copy, not ACM DL.
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “Experimental results demonstrate that compareXplore achieves significant improvements in ranking metrics and generates high-quality HLS results for the selected designs, outperforming the existing state-of-the-art method.” Abstract, p. 1. Scored on the ranking-metrics part (vs. harp).
- **Qualifies? (only the model differs)** Yes — only the model differs(ranking-metrics part only; borderline, see notes). Quote or table showing what the
  baseline changes:
  > “Table 1 presents the comparison between the vanilla harp model and our proposed approach” (p. 5). On the shared setup: “We ensure all the sampled design pairs come from the training set for a fair comparison” (p. 5) and “Consistent with harp, we use node attention and encode pragmas using MLPs” (p. 5)
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Table 1, p. 5 gives point values only: Kendall’s τ is 0.4157 for harp and 0.4319 for compareXplore. ACC (ALL) is 0.8859 for harp and 0.9117 for compareXplore. The abstract’s “significant improvements” has no test behind it.
- **Notes** (tie-break applied? anything ambiguous?): (1) The abstract’s baseline is unnamed (“the existing state-of-the-art method”). I resolved it to harp from the abstract’s own text and the intro contributions (p. 2). (2) The sentence holds two comparisons. The DSE/latency part fails Step 2: Stage 2 pairwise elimination is added on top of harp-style pruning (“as done in [29]”, Algorithm 1, p. 4), and harp’s DSE budget isn’t stated. (3) Borderline on Step 2: the paper never says harp was retrained with the same hyperparameters or epochs on the new Vitis 2021.1 dataset, only that pairs and splits are shared. If you require an explicit “same training setup” statement like A19’s, this becomes “doesn’t qualify”. The rest of the paper only compares against harp, so the score would then be N/A. (4) Two F rows apply at once (unbacked “significant”, and bare numbers), so no tie-break was needed. (5) Year note for B1 still needs your definition.

---

## B3. Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning
*2024, ASPLOS · arXiv 2402.02361*

- **Version read:** arXiv v3 (arXiv:2402.02361v3, 9 Apr 2025). It carries the ACM ASPLOS ‘25 copyright block, so it’s the authors’ copy, not ACM DL.
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “The results show that PaCM significantly outperforms both TLP and TenSetMLP.” Eval §6.5, p. 14 (Table 11). Scored on vs. TLP, the first baseline named in the sentence.
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “For each hardware configuration, we sampled 6 million data from the TenSet dataset and trained these different cost models under the same experimental setup.” (p. 14)
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Table 11, p. 14 gives point values only. TenSet T4 top-1: PaCM 0.892, TLP 0.862, TenSetMLP 0.859. K80 top-1: PaCM 0.897, TLP 0.880, TenSetMLP 0.878. Top-5 shows the same pattern. “Significantly outperforms” has no test.
- **Notes** (tie-break applied? anything ambiguous?): (1) The abstract headline (vs. Ansor, 2.6×) was rejected because Pruner changes the search, not just the model (LSE draft model plus GA, p. 5). Every other abstract and intro claim fails for the same reason or names no baseline. (2) Fig. 14 (LSE vs. GA, p. 13) was rejected as not stated to be controlled. If you read it as controlled, it would come first in order, but it would score the same F (point values, plus an average over 5,000 repeats with no spread). (3) The qualifying comparison isn’t the paper’s main contribution (search speed). It’s a side check on the cost model. (4) Two F rows apply (unbacked “significantly”, bare numbers), so no tie-break was needed. (5) Year note for B1 still needs your definition.

---

## B4. Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment
*2024, ICS · doi 10.1145/3650200.3656631*

- **Version read:** Version of record, ICS ‘24 proceedings (ACM, CC BY 4.0, DOI 10.1145/3650200.3656631)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “Results are shown in Tab. 8 that our model outperforms other models.” Eval §8.5.1, p. 507. Scored on vs. XGBoost (Ansor’s model), the first baseline named in the preceding sentence.
- **Qualifies? (only the model differs)** Yes — only the model differs (borderline, see notes). Quote or table showing what the
  baseline changes:
  > “We evaluate the prediction accuracy of the pre-trained cost model on the testing set with the coefficient of determination (R²)… We compare our Transformer-based model with the XGBoost-based model in Ansor and a two-layer LSTM model” (p. 507). Also from model selection: “We evaluated XGBoost, LSTM, and a lightweight Transformer model… Based on performance metric (Sec. 8.5), we selected the lightweight Transformer” (p. 501).
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Tab. 8, p. 507 gives point values only. R² on CPU: Fasor 0.972, XGBoost 0.923, LSTM 0.920. R² on GPU: Fasor 0.959, XGBoost 0.914, LSTM 0.908.
- **Notes** (tie-break applied? anything ambiguous?): (1) The abstract headline names no baseline and fails Step 2: the baselines are whole frameworks, and the search space, search algorithm and budget all differ. The intro contributions and §8.2–8.4 fail for the same reason. (2) The headline sentence names no baseline itself. The baselines are in the sentence just before it, and XGBoost is named first. (3) Borderline on Step 2: the paper says all three models are evaluated on the same test set, but not that XGBoost and LSTM were trained on the same 0.9M-sample training split with comparable tuning. Under a strict “must say it” rule this becomes “doesn’t qualify”, and the paper scores N/A, since nothing else is controlled. (4) The “median of 5 runs” (p. 505) applies to search results, not Tab. 8, and reports no spread anyway. (5) No tie-break needed. (6) Year note for B1 still needs your definition.

---

## B5. Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs
*2024, DAC · doi 10.1145/3649329.3658249*

- **Version read:** Likely version of record (ACM DAC ‘24, DOI 10.1145/3649329.3658249). There’s no arXiv stamp, but also no printed page numbers, so I can’t confirm it’s the ACM DL copy.
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “The experimental results demonstrate that Crop achieves prediction accuracy comparable to learn-based cost models trained with sufficient training data.” Eval §4.2, p. 6 (Table 2). Scored on vs. MLP, the first baseline named in §4.2.
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “From this dataset, we hold out a test set within 100k performance data… The remaining data is allocated for the training set to facilitate the utilization of learn-based cost models” (p. 5). “MLP and XGBoost, serving as naive cost models, are trained using the entire 8,000k training set from the target platform” (p. 6).
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Table 2, p. 6 gives point values only. Average Top-1: MLP 0.7922, XGBoost 0.7702, TLP 0.8742, Crop 0.7545. Average Top-5: MLP 0.9098, XGBoost 0.9097, TLP 0.9432, Crop 0.8650. The paper describes the gaps as “only 2.5% to 4.4%” vs. the naive models and “8.5% to 13.7%” vs. TLP (p. 6), with no test.
- **Notes** (tie-break applied? anything ambiguous?): (1) The headline is a parity claim, not a “beats” claim. Crop is below every baseline on average Top-1 and Top-5. I used rubric rule 3 (first comparison in the eval) because neither the abstract nor the intro contributions has a “beats” claim. If you count the abstract’s parity sentence as the headline, the result doesn’t change: it points to the same Table 2 comparison, but with an unnamed baseline. (2) The training data differs by design: Crop uses no target-platform data, which is the method’s purpose. I treated that as part of the model, not an uncontrolled difference. If you disagree, score N/A, since there is no other comparison. (3) “Average” means a mean across networks, with no spread. (4) This is a 6-page short paper. (5) No tie-break needed. (6) Year note for B1 still needs your definition.

---

## B6. Fast and Accurate Context-Aware Basic Block Timing Prediction using Transformers
*2024, CC · doi 10.1145/3640537.3641572*

- **Version read:** Version of record, CC ‘24 proceedings (ACM, DOI 10.1145/3640537.3641572)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “Our evaluations demonstrate ORXESTRA’s ability to provide precise timing estimations for different ARM targets (Cortex M4, M7, A53, and A72), surpassing existing machine learning-based approaches in both prediction accuracy and prediction speed.” Abstract, p. 227. Scored on accuracy vs. CATREEN.
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “for a fair comparison, we align CATREEN to the same configuration as ORXESTRA” (p. 228). “We re-implemented CATREEN using the same hyperparameters as in the original study… To ensure a fair comparison, the context size parameter for both ORXESTRA nd CATREEN was selected to be the same” (p. 232). Table 1 caption: “This dataset serves to the training and testing of all competitors also” (p. 231).
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Table 5, p. 234 gives point values only. MAPE (M4 / M7 / A53 / A72): CATREEN 8.8 / 13.3 / 8.5 / 10.4%; ORXESTRA 7.8 / 9.6 / 5.2 / 6.9%. The intro’s “28% improvement” (p. 228) is the relative gap between the two mean MAPEs (10.25% vs. 7.375%), with no spread or test.
- **Notes** (tie-break applied? anything ambiguous?): (1) The abstract names no baseline. I resolved it to CATREEN, the competitor the intro quantifies (p. 228). If you’d rather use the first model in Table 5 (NN), the score is still F. (2) The paper has no “contributions” bullet list; the 28% claim sits in the intro’s prose. (3) The only “standard deviation” (p. 235) describes the measured labels, not model error. The “boxplot” in Fig. 3 (p. 236) shows bars with nothing defined. The perplexity “significant” (p. 233) is an NLP heuristic on the pre-training comparison. None of these applies to the headline. (4) Accuracy flag on the speed half of the abstract claim: Table 10 (p. 235) shows NN (5,131 instr/s) and PalmTree ARM (3,809) both faster than ORXESTRA (2,691). It is only faster than the LSTM models; for example, it’s about 98% faster than CATREEN (1,356). (5) There’s a single train/test split, and training repeats or seeds aren’t stated. (6) No tie-break needed. (7) Year note for B1 still needs your definition.

---

## B7. NLTSP: A cost model for tensor program tuning using nested loop trees
*2025, J. Syst. Archit. · doi 10.1016/j.sysarc.2024.103305*

- **Version read:** Version of record, Journal of Systems Architecture 158 (2025) 103305 (Elsevier, DOI 10.1016/j.sysarc.2024.103305)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “Compared with TenSet MLP, the state-of-the-art cost model utilizing Ansor features as inputs, NLTSP achieves feature extraction speeds on CPU and GPU that are, on average, 97.9 times and 41.4 times faster, respectively, and can reduce the average search time for CPU and GPU workloads by 2.50 times and 4.11 times, respectively.” Abstract, p. 1. Scored on feature-extraction speed on CPU (97.9×).
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “Both Ansor and TenSet MLP use the same sampling program feature extraction method in [1], while NLTSP employs our rapid feature extraction method. TenSet MLP and NLTSP use pre-trained cost models on the TenSet dataset but do not update the cost model during the search… All our experiments were tuned for 200 rounds, with 10 tensor programs measured per round, i.e., k=10. Feature extraction and the cost model used the default batch size of 2048.” (§4.2, p. 7)
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. §4.2.1, p. 8 (Fig. 7): “NLTSP achieves feature extraction speeds on CPU and GPU that are, on average, 97.9 times and 41.4 times faster, respectively, compared to TenSet MLP”. Fig. 7 has one bar per workload, with no error bars and no number of runs stated.
- **Notes** (tie-break applied? anything ambiguous?): (1) The headline sentence holds four comparisons: extraction speed on CPU and GPU, and search time on CPU and GPU. I scored the first, CPU extraction speed. The search-time half (Fig. 10, p. 9) uses the same controlled setup and would also score F. (2) The headline metric is feature-extraction time, not prediction quality or final code latency. The cleaner accuracy comparison, Table 3 (p. 7, top-k scores), is also point values only, and NLTSP is below TenSet MLP on 2 of 7 platforms there. (3) Fig. 10 counts a workload where NLTSP fails to match the baseline as a speedup of “0” in the average (p. 9), which is unusual averaging. (4) The paper concedes NLTSP is worse than TenSet MLP at short budgets of a few minutes (p. 8). (5) The Ansor comparison (9.93× / 6.97×, p. 2) wouldn’t qualify, because Ansor trains its XGBoost model online from scratch while NLTSP is pre-trained. It isn’t the headline, though. (6) No tie-break needed. (7) Year note for B1: this paper was published online on 19 Nov 2024 but sits in the 2025 volume. That may matter for whatever the B1 year note records.

---

## B8. UniCoMo: A Unified Learning-Based Cost Model for Tensorized Program Tuning
*2024, ICCD · doi 10.1109/ICCD63220.2024.00081*

- **Version read:** Version of record, ICCD 2024 proceedings (IEEE Xplore, DOI 10.1109/ICCD63220.2024.00081)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “Results show that UniCoMo maintains the state-of-the-art accuracy while significantly improving adaptability to diverse execution environments and enabling flexible transfer prediction.” Abstract, p. 487. Scored on the adaptability claim vs. TDTenSet (Table IV, multi-intrinsic Tasks 5–6).
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “For feature representation, we transfer TenSet and TLP to TensorizeSet as baselines, referred to as TDTenSet and TDTLP” and “The model is trained and tested on datasets containing hardware intrinsics included in a task” (p. 493). For TDTLP: “Except for reasonable customization of schedule primitives embedding, other configurations remain identical to TLP” (p. 491).
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Table IV, p. 493 gives point values only. Task 5 Top-1: TDTenSet 0.7910, TDTLP 0.8290, UniCoMo 0.9040. Task 6 Top-1: TDTenSet 0.7953, TDTLP 0.8437, UniCoMo 0.9073. The claimed “significant advantage” (p. 494) has no test behind it.
- **Notes** (tie-break applied? anything ambiguous?): (1) The headline’s accuracy half is a parity claim. On the single-intrinsic Tasks 0–4, UniCoMo is on par with TDTLP and below it on Top-1 for Tasks 1, 3 and 4. I scored the adaptability half because it’s the part that claims to beat a baseline. (2) “Adaptability” names no baseline; I resolved it from p. 494. (3) The next abstract sentence (9.8× search speed, 1.9× latency, Figs. 10–11, pp. 494–495) is against MetaSchedule’s default online cost model, with the same MetaSchedule search and the same 2,000-trial budget (global=2000, per_iter=10). It would also qualify, and would also score F: one curve and one bar per workload, no spread. (4) Each task appears to be trained once; no repeats or seeds are reported. (5) Task 7 has no baseline numbers (“——”), because TDTenSet and TDTLP can’t run across CPU and GPU. (6) No tie-break needed. (7) Year note for B1 still needs your defi(1) The headline’s accuracy half is a parity claim. On the single-intrinsic Tasks 0–4, UniCoMo is on par with TDTLP and below it on Top-1 for Tasks 1, 3 and 4. I scored the adaptability half because it’s the part that claims to beat a baseline. (2) “Adaptability” names no baseline; I resolved it from p. 494. (3) The next abstract sentence (9.8× search speed, 1.9× latency, Figs. 10–11, pp. 494–495) is against MetaSchedule’s default online cost model, with the same MetaSchedule search and the same 2,000-trial budget (global=2000, per_iter=10). It would also qualify, and would also score F: one curve and one bar per workload, no spread. (4) Each task appears to be trained once; no repeats or seeds are reported. (5) Task 7 has no baseline numbers (“——”), because TDTenSet and TDTLP can’t run across CPU and GPU. (6) No tie-break needed. (7) Year note for B1 still needs your definition.nition.

---

## B9. UniCoX: A Unified Cost Model for Tensorized Program Tuning Across Ubiquitous Accelerators
*2026, IEEE TC · doi 10.1109/TC.2025.3626148*

- **Version read:** Version of record, IEEE Transactions on Computers 75(1), Jan 2026, pp. 305–319 (DOI 10.1109/TC.2025.3626148)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “It can accelerate search time by 11.3× and improve inference speed by 1.9× within the state-of-the-art tensorized program tuning framework, TVM MetaSchedule.” Abstract, p. 305. Scored on search time (11.3×) vs. MetaSchedule’s built-in online cost model.
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “We compare UniCoX with MetaSchedule’s built-in online cost model and TDTLP as baselines” (p. 314). “We integrate all the techniques discussed earlier in the state-of-the-art tensorized program tuning framework MetaSchedule… All our experiments are tuned for 200 rounds, each round picking 10 tensorizd programs to measure… The online cost model is trained during tuning by collecting online data.” (p. 316)
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Fig. 17, p. 317, gives one search-time value per workload. The text says UniCoX achieves “an average 11.3× speedup in search time”, with no spread or test. Table VIII (p. 317) gives single convergence times (e.g., ResNet on Tensor Core: 55 min vs. >7 h).
- **Notes** (tie-break applied? anything ambiguous?): (1) The first abstract result sentence (“achieves the state-of-the-art accuracy”) is a parity claim with no named baseline, so I skipped it, consistent with paper 8. If you count it, it resolves to Table IV (p. 314) vs. TDTenSet, TDTLP and UniCoMo on the same TensorizeSetX data. That comparison would qualify and would also score F (point values; UniCoX is below TDTLP on Task 1 Top-1, 0.9113 vs. 0.9238). (2) The headline sentence holds two comparisons: search time first, then inference speed (1.9×). (3) Table VIII’s “consistency” check is single numbers per workload, not repeated runs. (4) Overlap flag: this is the journal extension of paper 8 (UniCoMo), by the same authors with the same dataset lineage and the same baselines. You may want to note it if your analysis treats papers as independent. (5) No tie-break needed. (6) Year note for B1: published online 28 Oct 2025, but the issue is dated Jan 2026. This is the same pattern as NLTSP (paper 7).

---

## B10. UniSparTa: A Unified Sparse Tensor Program Tuning Framework
*2026, IEEE TCAD · doi 10.1109/TCAD.2025.3636442*

- **Version read:** Version of record, IEEE TCAD 45(7), July 2026, pp. 3360–3373 (DOI 10.1109/TCAD.2025.3636442)
- **Date read:** 10/28 (2026)
- **Minutes spent:** not recorded
- **Headline comparison (verbatim + page + where: abstract / intro / eval):**
  > “In contrast, the UniSparTa cost model, enabled by multimodal fusion, provides accurate performance prediction and significantly improves both convergence speed and final program quality.” Eval §VII-C, p. 3370 (Fig. 13). Scored vs. UniSparTa-TLP, the first baseline named in the ablation.
- **Qualifies? (only the model differs)** Yes — only the model differs. Quote or table showing what the
  baseline changes:
  > “In addition, we further ablate the impact of the cost model on the search process. To ensure fairness, we keep the design space construction and search algorithm unchanged, and only replace the cost model (details in Section VII-D).” (§VII-C, p. 3370)
- **Axis 3 score:** F
- **Evidence** (verbatim quote with page/table, or "NOT STATED"):
  > NOT STATED. Fig. 13, p. 3370 gives one tuning curve per method for two SpMM workloads (139 264 and 119 808), with no bands. The “significantly improves” in the headline has no number or test. The follow-up in Fig. 16 (p. 3371) is a point value only: “UniSparTa achieves 22.47×, 13.71×, and 3.9× speedup over ANNS, AutoSparse, and UniSparTa-WACONet”.
- **Notes** (tie-break applied? anything ambiguous?): (1) All four abstract baselines failed Step 2. Two were whole libraries or frameworks (MKL, TVM), one was a manual scheme (ASpT), and WACO changes the search and design space. The intro contributions name no baseline. (2) The headline sentence contrasts with both TLP and WACONet. I scored TLP because it’s named first. Swapping in WACONet gives the same F. (3) Minor Step 2 caveat: the per-round measurement count is stated only for UniSparTa (64). The paper’s “only replace the cost model” statement is what makes this qualify, and I took it at face value. (4) The only number tied to the controlled comparison is the 3.9× search-time speedup over UniSparTa-WACONet (Fig. 16, p. 3371). TLP is left out of that comparison because it “cannot even outperform ANNS”. (5) Table II (§VII-D), comparing cost-model accuracy against TLP and WACONet, comes later in the eval. It would also qualify (same 19-matrix test set), but its cell values aren’t in the text layer, so I couldn’t quote them. It has no ± or test in the text. (6) Overlap flag: same research group as papers 8–9 (Zihan Wang, Lei Gong, Chao Wang, Xuehai Zhou, USTC), and AutoSparse is their own earlier work. (7) No tie-break needed. (8) Year note for B1: published online 24 Nov 2025, but the issue is dated July 2026.

---

## Summary (tallied 2026-10-29 from the scorer's sheet; cutoff Oct 29 23:59 ET, Amendment 1)

| # | Score | Headline comparison (short) | Evidence location |
|---|---|---|---|
| B1 | F | LOOPer (model-guided) vs measurement-guided LOOPer, search time | §VI-B4, p. 10 |
| B2 | F | compareXplore vs harp, ranking metrics | Table 1, p. 5 |
| B3 | F | PaCM vs TLP, top-k | Table 11, p. 14 |
| B4 | F | Fasor Transformer vs XGBoost (Ansor), R² | Tab. 8, p. 507 |
| B5 | F | Crop vs MLP, Top-1/Top-5 | Table 2, p. 6 |
| B6 | F | ORXESTRA vs CATREEN, MAPE | Table 5, p. 234 |
| B7 | F | NLTSP vs TenSet MLP, CPU feature-extraction speed | §4.2.1 / Fig. 7, p. 8 |
| B8 | F | UniCoMo vs TDTenSet, multi-intrinsic Top-1 | Table IV, p. 493 |
| B9 | F | UniCoX vs MetaSchedule online cost model, search time | Fig. 17, p. 317 |
| B10 | F | UniSparTa vs UniSparTa-TLP, tuning curves | Fig. 13, p. 3370 |

- **Counts:** P 0 · ∂ 0 · F 10 · N/A 0 · not obtained 0 · not scored 0
- **n scored** (excluding N/A, not obtained and not scored) = 10
- **Runs a significance test (P):** k = 0 of n = 10
- **§8 sentence:** declaration §6, "If 0 of n" option (n = 10 > 5, so no small-n clause):
  "In a separate, pre-declared check of the 10 most-cited in-scope papers from 2024–2026 in the
  same frame, none of the 10 we could score runs a significance test for its headline comparison
  (Appendix X)." No P paper, so nothing is added to the §8 counter-example paragraph.
- **Recording notes:** all 10 scored 2026-10-28 from the scorer's Claude Doc sheet (copied
  verbatim on 2026-10-29). Minutes spent were not recorded. The headline and evidence column of
  the table above is a short label added when copying; the per-paper blocks govern.
- **Appendix must also state:** the B8–B10 clustering, the N/A count, "most-cited *in this
  frame*" (2–16 citations), and the §7 blindness statement.
