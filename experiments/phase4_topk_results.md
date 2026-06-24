# Phase 4 Results — The Downstream Top-K Autotuner Metric
## TPUGraphs GNN Compiler Optimization

**Status:** complete (5 matched seed pairs, K ∈ {1, 5, 10}).
**Headline:** ⛔ **Even on the Top-K metric a real autotuner uses, hardware edges give
no significant gain.** Third consecutive null — the downstream task did not rescue them.

> Companion to [`phase3_results.md`](phase3_results.md). Code: `phase4_evaluate_topk.py`.
> Ledger: [`../VERIFICATION.md`](../VERIFICATION.md) C8.

---

## 0. Why a different metric

Kendall's τ scores the *entire* ranking. But an XLA autotuner never uses the whole
ranking — it asks the model for its few best guesses, measures those on hardware, and
keeps the fastest. Two models with identical global τ could still differ at the extreme
**top** of the distribution, which is all the compiler cares about. Phase 4 asks: does
the global-τ null (Phase 3) hide a Top-K difference?

**Metric — Top-K slowdown** (lower is better; 1.0 = optimal). For each graph, with the
model's K highest-scored configs:

    slowdown@K = ( min actual runtime among the model's top-K )  /  ( min actual runtime overall )

slowdown@K = 1.20 means: an autotuner trusting this model's top-K ends up 20% slower
than the true-best config. Reported as the mean over the 7 validation graphs, then over
the 5 matched seed pairs (wide-baseline from Phase 2c vs hardware from Phase 3).

---

## 1. Results

| K | wide-baseline | hardware | Δ (hw − flat) | paired t-test p | hw better on |
|---|---:|---:|---:|---:|:---:|
| 1  | 1.214 ± 0.116 | 1.246 ± 0.054 | +0.033 | 0.65 | 2/5 seeds |
| 5  | 1.117 ± 0.056 | 1.084 ± 0.047 | −0.033 | 0.47 | 3/5 seeds |
| 10 | 1.092 ± 0.043 | 1.075 ± 0.038 | −0.017 | 0.62 | 3/5 seeds |

(error% = (slowdown − 1) × 100. Per-seed numbers in `phase4_evaluate_topk.py --all-seeds`.)

**Significance:** none. Every K has paired-t p > 0.46 and Wilcoxon p > 0.62. The
per-seed difference signs are mixed at every K (e.g. `+ − + − +` at K=1), so even the
*direction* is not consistent.

---

## 2. Interpretation

**No significant difference at any K — Phase 3's null holds downstream.** There is a
*faint directional hint* in the means (hardware slightly worse at top-1, slightly better
at top-5/10), which would be a tidy story — "hardware edges sacrifice the single best
pick but enrich the top handful." **But it is not significant, and the signs flip across
seeds.** Promoting that pattern to a claim would be precisely the cherry-picking we
committed not to do. Under the C9 noise floor (seed std here is 0.04–0.12 on the Top-K
slowdown), a mean Δ of 0.02–0.03 is indistinguishable from zero.

**Useful byproduct — task difficulty (architecture-independent).** Both models are
mediocre *top-1* autotuners: trusting only the single best-predicted config lands you
**~21% (flat) / ~25% (hw) slower than optimal**. You need the **top-10** to get within
**~7–9%**. This quantifies how far the models are from a deployable single-shot
autotuner, and is a fair, honest point for the paper independent of the comparison.

---

## 3. The complete picture (three nulls + one positive)

| Phase | Question | Metric | Result |
|------|----------|--------|--------|
| 2c | Does hierarchy help? | global τ | null (Δ=−0.012, p=0.22) |
| 3  | Do hardware edges help? | global τ | null (Δ=−0.012, p=0.72) |
| 4  | Do hardware edges help **at the top**? | Top-K slowdown | null (p=0.47–0.65) |
| — | **What *does* dominate?** | seed variance | **±0.04–0.12, ≫ every effect (C9)** |

The defensible contribution of this project is now fully formed: **two well-motivated
architectural inductive biases (hierarchical pooling, hardware-contention edges) fail to
beat a parameter-matched flat GNN — on both global ranking and the downstream Top-K
metric — because seed variance dominates the effect sizes on this dataset.** A
"Mythbuster" / negative-results + noise-floor paper, honestly earned.

---

## 4. Files

| File | Role |
|---|---|
| `phase4_evaluate_topk.py` | Top-K slowdown evaluator; `--seed N` (one pair) or `--all-seeds` (mean over 5) |
| `checkpoints/ablation/wide-baseline-seed*/` | flat baseline checkpoints (Phase 2c) |
| `checkpoints/phase3/hardware-seed*/` | hardware model checkpoints (Phase 3) |

---

## 5. Bottom line

> On the Top-K slowdown metric that an XLA autotuner actually uses, hardware-aware edges
> show **no significant improvement** over a parameter-matched flat GNN at any K
> (p = 0.47–0.65). The global-τ null is confirmed downstream. Both models leave ~21–25%
> on the table at top-1. There is no surviving evidence that the architectural
> interventions in this project help; the honest result is a characterized noise floor.
