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
- [ ] What EXACTLY is in `node_config_feat` (dims, semantics)? (gate for candidates 2/3)
- [ ] Does `reference/` expose per-config tile decisions, or only static shape? (gate for 3)
- [ ] What's the right "upper bound" definition — best achievable τ under an oracle, vs the
      information-theoretic ceiling?
- [ ] Runtime non-determinism control: can we measure runtime variance across near-identical
      configs to separate "complex-deterministic" from "noisy" (the C10 open question)?
- [ ] Compute budget + queue position (after 8c, before/after nlp_default resume?).
