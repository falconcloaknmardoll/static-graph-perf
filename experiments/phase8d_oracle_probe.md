# Phase 8d — Oracle / Upper-Bound Probe (design) — ⏳ STUB, FINALIZE WHEN 8C LANDS

> ⛔ **DO NOT run or finalize this yet.** Its exact form depends on Phase 8c's ending
> (results ~2026-07-13, `experiments/phase8c_results.md`). This stub exists so we DO NOT
> FORGET to design it. See PROJECT_TIMELINE.md "STRATEGIC REFRAME — 2026-07-11" and memory
> [[project-paper-spine]]. **Update trigger: the moment 8c's `.done` count = 20 and the
> evaluator has run.**

## Why this experiment exists (the one open reviewer hole)
The paper's last soft spot: *"convolution is just HARD to rank, not un-representable from the
static graph."* §7 shows no static feature tracks conv layout sensitivity — but that's a
*negative*. This probe converts it to a constructive positive: **show a model CAN rank conv
layouts once given the missing information**, proving the floor is **representational, not
fundamental**, and specifying what a faithful (differentiable) microarch performance model must
compute. This is also the flagship's constructive capstone (fixes "measures a floor, builds
nothing") and the SoP's grounding for the "build the representation that crosses it" agenda.

## The core idea (branch-independent)
Give the model **train-time-only, layout-config-specific information** that a static graph cannot
contain, and measure whether conv-subset ranking recovers toward an upper bound. If it does →
representational gap confirmed + the identity of the missing signal. If it does NOT → the failure
is deeper (label noise / genuine non-determinism) — also a publishable, sharper claim.

## Branch selection (decide after 8c)
- **If 8c gap CLOSES (Branch A):** the co-design features already partially cross the floor. 8d
  becomes the *upper-bound calibration* — how much of the achievable ceiling did the static MXU
  features capture? Oracle = ceiling; 8c features = fraction recovered.
- **If 8c gap PERSISTS (Branch B, pre-registered expectation P1/P3):** 8d is the *headline
  constructive result* — the oracle recovers ranking that static features could not.

## Candidate oracle signals (rank/verify feasibility when designing — DO NOT assume available)

> **Feasibility resolved 2026-08-17 (see "Open questions" below for detail):** candidate (1) is
> still the cleanest, unaffected by any of this. Candidate (2) is now **ruled out as stated** —
> `node_config_feat`'s 18 dims are already fully used, nothing richer exists in that key. Candidate
> (3) is **feasible but non-trivial** — the literal tile-size data exists, just in the separate
> `tile:xla` collection, and needs a real subgraph-matching step to join it onto layout-collection
> conv nodes before it's usable. This re-ranks the candidates: (1) cleanest/fastest to pre-register,
> (3) most physically meaningful but needs a scoping pass on the matching step first, (2) dead.
1. **Runtime-derived leakage probe (cleanest upper bound):** give the model a low-dimensional
   summary of the config's *own* runtime distribution statistics (e.g. per-config rank bucket
   from a held-out fold) → trivially high τ; the *point* is to bound achievable τ and show the
   task is learnable-with-the-right-info, framed honestly as an oracle, never as a real feature.
2. **Tiling/layout-decision signal:** the `node_config_feat` / layout minor_to_major already in
   the npz encodes the layout knobs — probe whether a model given *richer config-conditioned*
   features (beyond the 18-dim Early-Join) recovers conv ranking → the missing info is in the
   config encoding, not the base graph. ⚠️ verify what's actually in node_config_feat first.
3. **Per-config MXU tiling features** if recoverable from the config (im2col vs direct, spatial
   tile → MXU-pass count *at the chosen layout*, not the static shape) — the most physically
   meaningful, hardest to obtain. Check `reference/` featurizers for tile config availability.

## Method (skeleton — match Phase 8c rigor)
- Same frozen conv-vs-other split (phase8c_preregistration.md §4), same 5 seeds, paired.
- Arms: baseline (158) vs +oracle-signal; primary = conv-subset Top-1 slowdown, paired t +
  Wilcoxon + bootstrap CI; placebo on OTHER subset.
- **Pre-register 8d BEFORE running** (append an amendment or a new prereg) — same discipline.
- Honest framing everywhere: an oracle is an *upper bound / existence proof*, not a deployable
  model. Overclaiming here would be the exact sin the paper is about.

## Open questions to resolve at design time
- [x] **What EXACTLY is in `node_config_feat` (dims, semantics)? — RESOLVED 2026-08-17.**
      Fully documented in `reference/README.md` §"Layout Config Features". 18 dims, all already
      used (see `dataset.py`'s Early-Join, 140+18=158): dims 0–5 = output tensor physical layout
      (minor-to-major order, -1 padded if <6D), 6–11 = input tensor layout, **12–17 = kernel
      tensor layout, conv-ops only.** Confirmed on real data (inception_v3, 293/297 configurable
      nodes are conv nodes with a real kernel_layout): it genuinely varies per-config (2 distinct
      states observed across 4688 configs for a sampled node — a binary minor/major swap), it's
      not dead/padding. **Implication for candidate (2):** there is no "richer" field to add
      beyond the current 18 dims — `node_config_feat`'s full documented spec is already the
      complete 158-dim Early-Join baseline. Candidate (2) as originally framed ("richer
      config-conditioned features beyond the 18-dim encoding") **is not feasible from this key**;
      there's nothing more to extract from it. **Also resolves the Phase 8c config-invariance
      puzzle from a new angle:** the 158-dim baseline already sees real per-config conv-kernel
      layout info (dims 12–17) and still struggled to rank conv graphs — so "the baseline never
      sees per-config conv info" was never quite the right framing; the static MXU features
      (channel/spatial) added something the *existing* per-config signal didn't already supply.
- [x] **Does `reference/` expose per-config tile decisions, or only static shape? — RESOLVED
      2026-08-17, PARTIALLY YES, in a different collection than the one this project trains on.**
      `reference/README.md` §"Tile Config Features" documents a **separate 24-dim per-config
      vector** for the `tile:xla` collection (not `layout:xla`, which is what Phase 8c/all our
      training uses): dims 0–7 = literal kernel tile sizes (`kernel_bounds_0..5`, `_sum`,
      `_product`, conv-ops only), 8–15 = output tile sizes, 16–23 = input tile sizes (usually 0,
      inferable from the others). Verified locally: `data/predict-ai-model-runtime/npz_all/npz/
      tile/xla/valid/*.npz` exists, `config_feat` shape confirmed `(configs, 24)` matching the
      doc exactly. **This is the most literal version of candidate (3)'s signal** (actual tile
      sizes, not just layout ordering) — but **the tile collection is structurally different from
      layout**: its graphs are small per-kernel fused subgraphs (~18 nodes each, many files per
      source model), not the full per-model graphs the layout task uses, and **filenames don't
      correspond 1:1** (checked: 0 exact overlaps between `tile/xla/valid` and `layout/xla/random/
      valid`, though model-family prefixes match, e.g. both have `bert_pretraining`/`inception_v3`
      derivatives). **Using this for candidate (3) is feasible but requires a real mapping step**
      (matching tile-collection kernel subgraphs to specific conv nodes inside layout-collection
      full graphs) — not a drop-in feature, a genuine sub-task to scope before pre-registering.
