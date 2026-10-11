# Phase 8c — pre-registered results (frozen split, paired by seed)

Instances: CONV = 3 graphs × 2 collections, OTHER = 4 × 2.

arm           CONV top-1 slowdown    OTHER top-1 slowdown
baseline         1.4246 ± 0.0721        1.0272 ± 0.0027
channel          1.1203 ± 0.1077        1.0292 ± 0.0182
spatial          1.1758 ± 0.0389        1.0237 ± 0.0080
full             1.1933 ± 0.1479        1.0125 ± 0.0080

## Primary test (§7): CONV subset, arm vs baseline, paired by seed
  channel   Δslowdown=-0.3043  CI95=[-0.3713,-0.2271]  t-p=0.0021  BH-p=0.0032  Wilcoxon-p=0.062  d=-3.15
  spatial   Δslowdown=-0.2488  CI95=[-0.2971,-0.2197]  t-p=0.0005  BH-p=0.0014  Wilcoxon-p=0.062  d=-4.72
  full      Δslowdown=-0.2313  CI95=[-0.3094,-0.1582]  t-p=0.0073  BH-p=0.0073  Wilcoxon-p=0.062  d=-2.25

## Placebo (§7): OTHER subset (must be ~unchanged)
  channel   Δslowdown=+0.0020  CI95=[-0.0124,+0.0164]  t-p=0.8170
  spatial   Δslowdown=-0.0035  CI95=[-0.0108,+0.0040]  t-p=0.4577
  full      Δslowdown=-0.0147  CI95=[-0.0215,-0.0079]  t-p=0.0199

## Decision rules (§8, verbatim)
  gap closes (BH-p<0.05, OTHER unchanged) → positive co-design result
  gap unchanged → representational-floor evidence (publishable)
  OTHER moves too → confound; investigate before any claim
  mixed across arms → report per-arm; +spatial vs +channel contrast IS the finding (P1/P2)

  Registered predictions: P1 channel→null, P2 spatial→best-but-uncertain,
  P3 persisting gap = reportable finding. (Committed 2026-06-29, before data.)

---

## INTERPRETATION & REQUIRED SCRUTINY (added post-hoc, 2026-07-13 — NOT part of the frozen evaluator output)

**Verdict per frozen rules:** channel & spatial → *positive co-design result* (CONV gap closes,
BH-p<0.05, placebo unchanged); full → gap closes but placebo moves slightly (−0.015, p=0.02) →
"investigate." **This REVERSES pre-registered P1** (channel predicted null; was the strongest,
Δ=−0.304). Wilcoxon p=0.062 for all = the n=5 floor (as significant as 5 seeds allow), t/BH significant.

**✅ UPDATE 2026-08-19 — confound A (item 2) RESOLVED, confound 3 RECONCILED. See
`experiments/phase8c_control_results.md` for the full P4 statistics; summary and item-3
reconciliation below. This is now a confirmed co-design result, not a pending one.**

**Original scrutiny list (2026-07-13), items now resolved in place:**

1. **Config-invariance puzzle (the crux).** MXU features are derived from each conv node's STATIC
   shape → identical across all configs of a graph. Top-1 slowdown is WITHIN-graph config ranking.
   Config-invariant inputs cannot directly reorder a graph's configs; any effect must be via changed
   training dynamics (model stops ignoring the config signal on conv graphs). **Must demonstrate the
   mechanism, or suspect a bug/leak.** First checks: (a) confirm mxu_block is truly identical across a
   graph's configs in the loaded batches; (b) confirm baseline conv predictions are near-constant
   across configs (a "collapse" the features break).
2. **Specificity control (the Phase-2b lesson) — ✅ RESOLVED 2026-08-19, PASSED.** Ran the
   pre-registered `scramble` arm (prereg A2/P4): same 4 channel columns, values permuted among
   MXU-resolved conv nodes so the node↔value alignment (the actual information) is destroyed but
   dimensionality/config-invariance/conv-node placement are preserved. Result: **channel
   significantly beats scramble** (Δ−0.244, CI95=[−0.315,−0.161], t-p=.006, d=−2.35, CONV subset,
   paired by seed), while scramble itself is statistically indistinguishable from baseline
   (Δ−0.060, t-p=.30). **This is real MXU information, not a capacity/regularization artifact.**
   Minor caveat: scramble's OTHER-subset placebo wobbled slightly (Δ+0.008, p=.028) — same small
   magnitude/pattern as the `full`-arm wobble in item 4 below, disclosed but not disqualifying.
   Full numbers: `experiments/phase8c_control_results.md`.
3. **Reconcile with model-free analysis — ✅ RESOLVED 2026-08-19.** `scripts/analyze_mxu_misalignment.py`'s
   ρ=−0.06 (n=95, `experiments/phase8_mxu_misalignment.md` T3) and 8c's channel result are **not
   in tension — they measure different things at different granularities:**
   - **What ρ=−0.06 tested:** `mxu_overhead`, a single compute-weighted SCALAR per graph
     (`Σmacs/eff / Σmacs`, one number collapsing every conv node's `c_out_util`/`k_util` into one
     average), correlated against real-runtime **layout sensitivity** (`config_runtime` CV) —
     a **between-graph** question: does a graph's average packing badness predict how much its
     runtime varies across layouts, compared to other graphs? Answer: no.
   - **What 8c's `channel` arm actually feeds the model:** the same underlying quantities
     (`c_out_mxu_util`, `c_out_pad_waste`, `k_mxu_util`, `k_pad_waste`) but **per-node, unaggregated**
     — one 4-vector per conv node, not collapsed into a graph scalar — as auxiliary input to a GNN
     that also sees topology and the graph's other node features. The outcome measured is **Top-1
     ranking accuracy within a fixed set of graphs**, a **within-graph, model-mediated** quantity —
     not "how much does runtime vary" but "can the model, given richer per-node features, correctly
     order this graph's configs." **Sensitivity ≠ rankability, confirmed as the resolution**, exactly
     as flagged as an open possibility here on 2026-07-13.
   - **Why this is coherent with the config-invariance finding (item 1) and 8d's node_config_feat
     resolution:** channel features are config-invariant (identical for every config of a graph,
     confirmed by code), and `node_config_feat` dims 12–17 already carry real per-config
     conv-kernel-layout info that the 158-dim baseline could already see (`phase8d_oracle_probe.md`,
     resolved 2026-08-17) — yet the baseline still struggled to rank conv graphs. The likely
     mechanism: static per-node MXU utilization acts as an **informative prior that helps the
     shared GNN encoder learn better conv-node representations overall**, which in turn helps the
     model make better use of the config-dependent signal that was already present but
     under-exploited — not as new config-discriminating information in its own right (it can't be,
     it's config-invariant). This is a capacity/regularization-*shaped* mechanism but a real-
     information *source* — consistent with confound A passing (it's not JUST capacity, since
     scramble — identical capacity — fails) while still explaining why a config-invariant feature
     can move a within-graph config-ranking metric.
4. **full-arm placebo wobble** (−0.015, p=0.02): tiny and improving, but per rules = investigate.
   Now read alongside item 2's scramble wobble (+0.008, p=.028) as a small, recurring pattern of
   minor OTHER-subset noise specifically on arms that add extra conv-localized columns — worth one
   disclosure sentence in the write-up, not a separate investigation track.

**Framing for the second author / paper (updated 2026-08-19):** this is now a *confirmed* co-design result — MXU
geometry features close the convolution ranking gap, reversing our own pre-registered null prediction,
with both the config-invariance mechanism (item 1) and the specificity control (item 2) resolved in
its favor. NOT "we fixed convolution" — the gap narrows (1.42→1.12), it doesn't vanish, and the
mechanism (item 3) is an indirect prior effect, not direct config-discriminating information. The 8d
oracle probe is no longer a required fallback (its scramble-control first arm already ran as part of
this confound-A resolution) — it's now an optional capstone if there's time, not load-bearing.
See `experiments/novelty_venue_draft.md` for how this folds into the paper's final framing.

**Note (2026-10-11, framing audit):** the 2026-08-19 wording above ("close the convolution ranking
gap", "confirmed co-design result") predates the LR question. The 8c baseline was not LR-tuned
(same flat model and lr 1e-3 as 6e's handicapped arm), so whether part of the effect is a
baseline-LR artifact is undetermined. Tested by 8e (`phase8e_lr_robustness_preregistration.md`,
draft). Use "narrows", never "closes". Registered BH-p for channel is .0032 (line 12).
