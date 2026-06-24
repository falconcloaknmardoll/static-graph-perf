# Verification Ledger — Human-in-the-Loop Pass

**Purpose.** A large fraction of this project was produced with AI assistance. For the work
to be honestly publishable (ICML / ASPLOS / AISTATS) and genuinely *yours*, every load-bearing
claim and every load-bearing piece of code needs a **human pass** — you personally re-deriving,
re-running, or critically checking it, not just trusting that it looks right.

This file is the ledger. Each item has a **status**:

- 🔴 `AI-asserted` — produced by AI, **not yet** independently verified by a human. Do not put
  in the paper as-is.
- 🟡 `partially-checked` — some human review done; gaps remain (noted).
- 🟢 `human-verified` — you have personally confirmed it and can defend it under questioning.

**How to use:** when you verify an item, change its status, add the date, and note *how* you
checked it. The goal is to drive everything load-bearing to 🟢 before submission.

> Why keep this at all? Reviewers (and your own integrity) reward a project that *knows* what it
> has and hasn't established. An explicit "here is what is verified vs. asserted" record is a
> strength, not an admission of weakness.

---

## A. Claims (the things the paper would assert)

| ID | Claim | Status | What a human must do to promote it |
|----|-------|--------|-------------------------------------|
| **C1** | "We beat the TPUGraphs paper baseline of τ=0.34." | 🔴 AI-asserted | Confirm our τ is computed the **same way** as the paper: same `layout:xla:random` split, same per-graph-then-averaged protocol, same tie handling. Then report on the **official held-out test set**, not just the 7 val graphs. |
| **C2** | ~~"Hierarchical pooling (DiffPool) is what causes the gain over the flat baseline."~~ **REFUTED** | 🟢 human-verified (negative) | **Resolved 2026-06-22 by the 5-seed parameter-matched ablation** ([`experiments/phase2c_ablation_results.md`](experiments/phase2c_ablation_results.md)). At 236k params for both arms: wide-baseline τ=0.2914±0.0343 vs diffpool τ=0.2797±0.0259, **Δτ=−0.0117, paired t-test p=0.22, Wilcoxon p=0.31 → no significant difference** (flat numerically ahead, 3/5 seeds). The Phase 2b gap was a parameter-count artifact. The claim is **false as stated**; do not put "hierarchy causes the gain" in the paper. |
| **C3** | ~~"The gain is due to mitigating over-squashing (low spectral gap / large diameter)."~~ **UNSUPPORTED** | 🔴 contradicted | The over-squashing mechanism predicted hierarchy (diameter reduction) should beat a same-size flat model. C2's ablation shows it does **not** → the data now argue *against* this mechanism at this scale. Drop the causal over-squashing claim, or reframe as an open question. If you still want to test it directly: correlate per-graph τ with graph diameter, or measure representation collapse in the flat model — but the headline evidence is currently negative. |
| **C4** | "τ = 0.3676 (GraphSAGE warm) / 0.3560 (DiffPool cold) are real, reproducible results." | 🟡 partially-checked | **Now have 5-seed error bars for the param-matched ablation arms** (wide-baseline 0.2914±0.034, diffpool 0.2797±0.026 — note these are at the 128-config cap, *not* the same regime as the 0.3676/0.3560 headline runs). Seed-to-seed std is **~0.03**, large relative to the gaps we were treating as meaningful — strong reason to distrust any single-run comparison. **Still TODO:** ≥3 seeds for the *headline* configs (GraphSAGE warm at 512 cap) before 0.3676 goes in the paper as a number. |
| **C5** | "Graphs are compute-dominated (~63%), motivating the approach." (Phase 0.5) | 🔴 AI-asserted | This number used the **wrong opcode enum** (see code-audit V5). Re-derive with the corrected map from `scripts/extract_opcode_map.py` before it motivates anything. |
| **C6** | "Warm-start helps GraphSAGE but hurts DiffPool (fragile multi-loss equilibrium)." | 🟡 partially-checked | The data (logs) clearly show the directional effect. But it is **n=1 each**; fold into the C4 multi-seed runs to confirm it is not a single-seed artifact. |
| **C7** | "DiffPool clusters stay soft / don't collapse (entropy rises, LP falls)." | 🟡 partially-checked | Supported by the loss-equilibrium figure (real logged data). To fully back it, **inspect an actual assignment matrix S** on a held-out graph and confirm it is genuinely distributed, not just that the scalar entropy rose. |
| **C8** | ~~"Hardware-aware contention edges improve ranking over the flat baseline."~~ **REFUTED** | 🟢 human-verified (negative) | **Resolved 2026-06-23 by the Phase 3 5-seed run** ([`experiments/phase3_results.md`](experiments/phase3_results.md), same seeds/regime as C2). hardware τ=0.2798±0.0396 vs wide-baseline 0.2914±0.0343, **Δτ=−0.0116, paired t-test p=0.72, Wilcoxon p=0.63 → no significant difference** (flat ahead 3/5). Model verified correct (sparse `Sᵀ Eᶜ S` matches dense ref to 1e-7), so it's a genuine null, not a bug. Do **not** claim hardware edges help on global τ. **Phase 4 update (2026-06-23):** the downstream **Top-K** metric also shows no significant difference (K=1/5/10 paired p = 0.65/0.47/0.62; mixed signs) — see [`experiments/phase4_topk_results.md`](experiments/phase4_topk_results.md). The null holds on the metric an autotuner actually uses. |
| **C9** | "Seed variance dominates all architectural effects on this dataset." | 🟢 human-verified | Established across C2 + C8: paired-difference std is ≈0.04–0.07, ~6× the largest mean effect we measured. Same seed swung ±0.09 for the *same* architecture change. **Implication for the paper:** every τ comparison below ~0.05 needs multi-seed error bars or it is uninterpretable. This is a load-bearing *positive* finding (the one thing we can assert confidently). |
| **C10** | "Cost models fail on convolution-dominated graphs (~47% off Top-1 vs ~5% on transformer/UNet)." | 🟡 exploratory (do NOT confirm) | Phase 5 ([`experiments/phase5_error_analysis.md`](experiments/phase5_error_analysis.md)): perfect 3-vs-4 separation, Mann-Whitney p=0.029 (the *floor* for 3-vs-4), top correlate `mxu_frac` ρ=+0.68 p=0.09. **Three reasons it is NOT a finding:** (1) CNN-vs-other grouping is **post-hoc** (HARKing risk); (2) n=7, 9 features, no multiple-comparison control, "worst 10%"=1 graph; (3) "non-deterministic" is unproven — static features failing to *rank* conv configs is equally consistent with a *complex-but-deterministic* layout map. **To promote:** pre-registered split on the official/larger test set + a direct runtime-variance (non-determinism) measure. Strong *hypothesis*, not a result. |

---

## B. Code correctness (the enum bug proved AI code here can be silently wrong)

Trace each of these yourself, by hand, against a tiny example — do not assume.

| ID | Component | Status | What to check |
|----|-----------|--------|---------------|
| **V1** | ListMLE loss (`loss.py`) | 🔴 AI-asserted | Confirm it implements the listwise likelihood correctly (the `logcumsumexp` ordering, normalization by K). Hand-check on a 3-item list with a known correct ordering. |
| **V2** | τ sign convention (`train.py` `evaluate`) | 🔴 AI-asserted | We compute `kendalltau(preds, −runtimes)`. Verify the minus sign is correct: a model that perfectly predicts "low runtime = high score" must yield τ → +1, not −1. |
| **V3** | Early-Join scatter (`dataset.py`) | 🟡 partially-checked | `test_early_join_integrity` covers this, but confirm the config features land on the **correct node rows** (`node_config_ids`) and the ×100 scaling is intended. |
| **V4** | DiffPool sparse `A@S` (`model_phase2.py`) | 🔴 AI-asserted | The `scatter_add` trick replaces a dense N×N matmul. Verify on a **tiny graph** that the sparse result equals the dense `A @ S` exactly. |
| **V5** | Opcode enum (`analyze_opcodes.py`) | 🟡 partially-checked | Corrected to TPUGraphs custom enum (2026-06-21). MEMORY_BOUND_OPCODES and COMPUTE_BOUND_OPCODES now match `scripts/extract_opcode_map.py`. **Still needed:** re-run `analyze_opcodes.py` and confirm the "~63% compute-dominated" headline number changes to the correct value (feeds C5). |
| **V6** | Phase 3 structural edges (`dataset.py`) | 🟢 human-verifiable now | `test_structural_edges` asserts shape/one-hot/bound; the verification run (BERT now has Dots) confirms the opcode fix. Spot-check one graph's edges by hand to reach 🟢. |

---

## C. Methodology gaps (what a strong reviewer will ask for)

- **Seeds & significance** (drives C4, C6) — ✅ *partially closed* by the Phase 2c 5-seed
  ablation, which establishes seed std ≈ 0.03. Remaining gap: the **headline** configs
  (GraphSAGE warm, 512-cap) still have no error bars.
- **Ablations** (drives C2, C3) — ✅ *parameter-matched baseline done* (Phase 2c) and it came
  back **negative for hierarchy**. Still open if pursuing DiffPool further: remove aux losses one
  at a time; vary pooling ratio. But the headline ablation already answers C2.
- **Test-set evaluation** (drives C1) — 7 val graphs is too few for headline numbers.
- **Compute/latency honesty** (for ASPLOS especially) — if claiming a *systems* contribution,
  characterize training/inference cost, not just τ.

---

## How to make this *your* work

Pick claims in priority order. **C2 (param-matched baseline) is now done — and it overturned the
hierarchy claim** (Phase 2c). That is the model to follow: a controlled experiment beats an
assertion every time, even (especially) when it kills a result you liked. Next priorities:
**C1 (eval protocol + test set)** and **C4 for the headline GraphSAGE-warm number**. For each:
re-run it yourself, read the code that produced it until you could re-implement it from scratch,
and only then flip the status to 🟢.

> **Process note for the human pass:** the AI-produced Phase 2b write-up confidently claimed
> hierarchy was "proven." It was not — it was a parameter-count confound, and only a controlled
> ablation exposed it. Treat every remaining 🔴 with the same suspicion.

_Last updated: 2026-06-22 — C2 resolved (negative), C3 contradicted, C4 partial. Update as you go._
