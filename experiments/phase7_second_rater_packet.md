# Phase 7 — Second-Rater Packet (blind scoring, for Cohen's κ)

> **FILLED 2026-08-19 — ⚠️ IMPORTANT CAVEAT ON WHAT THIS IS.** This packet was scored by a
> fresh, context-isolated AI agent (no exposure to rater-1's scores or this project's prior
> conversation) rather than by a human. It IS a genuinely blind independent pass — the two
> scoring runs were done without seeing each other's answers — but it does **NOT** fully satisfy
> `phase7_meta_analysis.md` §D's "two independent raters" intent, which (per that doc's own
> "Method & anti-fabrication guardrail" §, item 3) exists specifically because "AI-prescreened"
> scores need "a human confirming the quotes" to become "verified." Two AI passes, even
> genuinely blind ones, are correlated in ways a human-vs-AI check would not be (same kind of
> failure modes, same training-data blind spots). **Treat the κ below as a real, useful pilot
> signal — not as license to quote a headline % as satisfying the paper's own rigor bar.** A
> human pass on this same 5-paper sample (or a fresh larger one) is still the right next step
> before citing anything from Phase 7 in the paper.

**Purpose.** `experiments/phase7_meta_analysis.md` §D requires "two independent raters on a
≥20% sample; report Cohen's κ" before any headline percentage from the literature audit can be
quoted. This packet is that independent pass. **Do not open
`experiments/phase7_rater1_scores_SEALED.md` until you have scored all 5 papers below** — it
holds the first rater's (AI-assisted) scores, and seeing them first would defeat the point of an
independent check.

**Sample.** 5 of the 20 papers scored so far (25% — clears the ≥20% bar), chosen by
`random.seed(20260819)` (today's date, documented here so the selection is verifiably not
cherry-picked) over the full A1–A20 list. See bottom for the exact selection code.

**How to score.** Read `experiments/meta_analysis_rubric.md` first (the 5 axes + P/∂/F anchors)
— it's short, ~1 page. Then read each paper below (arXiv link given; PDF or HTML both fine) and
score it P/∂/F on each axis **from your own read of the text**, independent of any other
document in this repo. Write 1-2 sentences of justification per axis (a quote from the paper is
ideal but not required — this is scoring for agreement, not producing a citable audit entry).

**Papers to score:**

1. **ProGraML** (Cummins, Fisches, Ben-Nun, Hoefler, Leather; ICML 2021) — arXiv 2003.10536
2. **TenSet** (Zheng, Liu, Shao, Chen, Gonzalez, Stoica, Haj-Ali; NeurIPS D&B 2021) — no arXiv
   ID; NeurIPS Datasets & Benchmarks proceedings (search "TenSet NeurIPS 2021 proceedings")
3. **Facile** (Abel, Sharma, Reineke; IISWC 2023) — arXiv 2310.13212
4. **Regression Language Models for Code** (Akhauri, Song, Wongpanich, Lewandowski, Abdelfattah;
   Google, 2025) — arXiv 2509.26476
5. **HLS Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing** (Wu, Yang,
   Xie, Li, Hao; DAC 2022) — arXiv 2201.06848

**Scope reminder (from the rubric):** only score the paper's *comparative architectural claim*
("architecture/feature X improves ranking/accuracy over baseline Y"). If a paper makes no such
claim, mark N/A and say why — but all 5 papers above were pre-screened as in-scope, so this
should not come up.

---

## Scoring sheet — fill in below

### 1. ProGraML — arXiv 2003.10536
| Axis | Score (P/∂/F) | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | F | No seed count, no mean/std/CI anywhere in Tables 3/5/6; the only repetition is 10-fold CV over data splits for device mapping, not re-initialization. |
| 2 Param-matched | F | Only the Transformer baseline's size is given ("5.6 million trainable parameters"); ProGraML/GGNN, NCC, DeepTune, DeepTuneIR parameter counts are never reported or matched. |
| 3 Significance test | F | No p-values, CIs, or hypothesis tests anywhere; Tables 3, 5, 6 are raw point estimates. |
| 4 Eval-set integrity | ∂ | "We divided the datasets randomly using a 3:1:1 ratio for training, validation, and test" (Case A) but exact test-set size never stated; Case B test folds are only ~68 instances; Case C states an explicit held-out ~10,000 test set — mixed rigor across the three case studies. |
| 5 Effect vs noise | ∂ | Case A: "3.44× improvement" (large, variance unknown); Case B gain is only ~6.3 pts on AMD / ~1.5 pts on NVIDIA vs NCC, zero variance reported. |

### 2. TenSet — NeurIPS D&B 2021
| Axis | Score (P/∂/F) | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | F | Table 4 and Fig. 6a report single numbers per cell/bar; no seed count, std, or CI stated anywhere in the main text. |
| 2 Param-matched | F | "Further training details are included in Appendix C" (not in main text); no parameter counts or capacity control given for MLP vs LSTM vs GRU vs GraphSAGE vs GCN vs XGBoost. |
| 3 Significance test | F | No hypothesis test anywhere; Table 4 top-k scores bolded as "highest" with no test of margin significance. |
| 4 Eval-set integrity | ∂ | "We hold out a test set that consists of five networks... and train models... on the training set that consists of the whole dataset excluding all the tasks that appear in the above five networks" — explicit held-out set, but no separate validation split described for model selection. |
| 5 Effect vs noise | F | Margins between MLP+Ranking (0.8823 top-1) and next-best (XGBoost+MSE 0.8535, LSTM+MSE 0.8637) are 0.02-0.03 with zero variance reported; the 10× search-time claim is also single-run per bar. |

### 3. Facile — arXiv 2310.13212
| Axis | Score (P/∂/F) | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | F | Ithemal used off-the-shelf with no retraining/seeds; "We could not experimentally evaluate GRANITE as the trained model is not publicly available" — GRANITE numbers are cited, not run, no variance data. |
| 2 Param-matched | F | Facile is hand-built analytical (uarch parameters from uiCA / uops.info) vs learned Ithemal/GRANITE; no attempt to quantify or control complexity/tuning effort on either side. |
| 3 Significance test | F | No p-values or tests; Table 2 differences (e.g. 0.42% vs 0.49% MAPE) asserted from raw numbers only. |
| 4 Eval-set integrity | F | GRANITE claim: "based on the results reported in the paper... Facile significantly outperforms GRANITE on all microarchitectures" — a cross-paper metric borrow, not a shared eval run; even the direct BHiveU/BHiveL comparison never discloses filtered eval-set counts. |
| 5 Effect vs noise | ∂ | Facile vs uiCA near-tied (~0.1-0.5% MAPE, no variance given); Facile vs Ithemal is large (8.28% vs ~1%) but Ithemal used unmodified rather than freshly tuned for newer microarchitectures. |

### 4. Regression Language Models for Code — arXiv 2509.26476
| Axis | Score (P/∂/F) | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | F | Appendix lists hyperparameters but no seed count; Table 4 (NAS-Bench Kendall-τ) has no error bars, std, or CI. |
| 2 Param-matched | F | RLM stated as "a relatively small 300M parameter RLM based on T5Gemma," but Arch2Vec, CATE, GNN, and FLAN parameter counts are never given or matched. |
| 3 Significance test | F | No hypothesis test for any NAS-Bench comparison; Table 4 is raw Kendall-τ. |
| 4 Eval-set integrity | ∂ | Sample counts matched to prior work ("16 samples... for NASNet, Amoeba, PNAS and 100 samples for DARTS to match FLAN settings"), but train/test architecture counts and any separate validation split are never stated. |
| 5 Effect vs noise | F | vs FLAN (paper's own prior SoTA): average Kendall-τ gain +0.002 (0.461 vs 0.459), sign-flipping across sub-benchmarks (PNAS -0.003, ENAS -0.003, DARTS -0.039), zero variance reported — sub-noise. |

### 5. HLS-GNN Benchmarking — arXiv 2201.06848
| Axis | Score (P/∂/F) | Evidence (quoted) |
|---|---|---|
| 1 Seeds/variance | ∂ | "Each model is trained with five runs using different random number seeds and we report the average of three with least validation error" — multiple seeds, but best-of-N selection with no dispersion (std/CI) reported. |
| 2 Param-matched | ∂ | All 14 GNNs share hidden-dim=300 / 5 layers, but actual parameter counts (which differ by aggregation mechanism, e.g. GAT's attention weights) are never reported or matched. |
| 3 Significance test | F | No p-values or hypothesis tests; Tables 2/4/5 are raw MAPE percentages. |
| 4 Eval-set integrity | ∂ | "DFG and CDFG datasets are randomly split into 80% train, 10% validation and 10% test" — sound protocol, but exact test-set counts are never explicitly stated (only inferable from the ~40k total). |
| 5 Effect vs noise | ∂ | Large spread between best/worst GNN and vs Vitis HLS baseline, but the IronMan comparison appears cross-paper/inferred rather than a direct shared-eval table, and no variance/significance backs any of it. |

**Rater agreement with the sealed rater-1 scores: 21/25 = 84% raw, Cohen's κ ≈ 0.67 (pooled
across all 5 axes × 5 papers) — see `phase7_meta_analysis.md` §D for the full breakdown.**

---

**When done:** hand this filled-in sheet back. Cohen's κ per axis (and overall) will be computed
against the sealed rater-1 scores and reported in `phase7_meta_analysis.md` §D.

**Selection code (for reproducibility):**
```python
import random
papers = [  # A1..A20, in phase7_meta_analysis.md order
    'A1. TPUGraphs', 'A2. Cross-attention TGraph', 'A3. GRANITE', 'A4. Ithemal',
    'A5. ProGraML', 'A6. TLP', 'A7. PerfSAGE', 'A8. NeuSight', 'A9. Kaufman-TPU',
    'A10. AutoTVM', 'A11. TenSet', 'A12. Polynormer', 'A13. Reconfigurable Dataflow HW',
    'A14. Facile', 'A15. COGNATE', 'A16. Regression LMs for Code', 'A17. Omniwise',
    'A18. HLS-GNN Benchmarking', 'A19. Toward Compiler World Models', 'A20. PerfVec',
]
random.seed(20260819)
sample = random.sample(papers, 5)  # -> A11, A14, A16, A18, A5
```
