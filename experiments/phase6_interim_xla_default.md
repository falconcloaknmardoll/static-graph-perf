# Phase 6 INTERIM — xla:default breaks the null (PRELIMINARY, sweep still running)

> **Status: PRELIMINARY (2026-07-07).** xla/default is complete (10/10 runs); nlp:random is
> 7/10; nlp:default not started. No narrative changes to the paper until the sweep finishes
> AND the disaggregation ablation below is run. This doc records the result + the confound
> analysis so we don't repeat the Phase-2b mistake (attributing a win to the wrong component).

## The result (5 matched seeds, paired, identical regime to Phase 2c/3)

**xla:default — the null does NOT replicate. The hardware-aware hierarchical model wins
decisively and consistently:**

| seed | flat τ | hardware τ | Δ (hw−flat) |
|---|---|---|---|
| 42 | +0.0269 | +0.1930 | **+0.1661** |
| 100 | +0.0310 | +0.2118 | **+0.1808** |
| 123 | +0.0213 | +0.1959 | **+0.1746** |
| 456 | +0.0285 | +0.1788 | **+0.1503** |
| 789 | +0.0561 | +0.2090 | **+0.1529** |

- flat τ = **0.0327 ± 0.0135** (near-floor); hardware τ = **0.1977 ± 0.0133**
- paired Δτ = **+0.1650 ± 0.0133** — paired t **p < 0.001**; Wilcoxon p = 0.062 (the n=5
  floor: all 5 seeds same direction). Effect ≈ **12× the paired seed-std** — an order of
  magnitude above the C9 noise floor (0.04–0.07 on xla:random).

**Context that makes it credible rather than absurd:**
- TPUGraphs paper (Table 2, GraphSAGE baseline): xla:default **val τ = 0.12** vs
  xla:random val 0.19 / test 0.34 → xla:default is *known-much-harder*; a weak flat model
  there is consistent with the reference. (Our flat 0.033 < their 0.12 — plausibly our
  128-config cap / 50 epochs; both arms share the regime, so the *comparison* is internally
  valid. ⚠️ external-validity check later.)
- Training logs: flat's loss barely moves (1.315→1.291, val_τ noise ~0.02–0.06 all run) —
  it genuinely fails to learn the task; hardware's lm-loss declines steadily with val_τ
  climbing 0.17→0.19 — real learning signal, not eval luck.
- Our hardware model (0.198 val) **exceeds the paper's own baseline** (0.12 val) on this
  collection.

**nlp:random partial (3/5 pairs):** flat 0.62–0.78 (task much easier), Δ = +0.002, −0.145,
−0.068 → null-to-flat-leaning. So the xla:default effect is **collection-specific**, not
a universal hardware-model win. Also note: preliminary nlp flat seed-std looks ~0.08 at
n=3 — the "more val graphs tightens the floor" hypothesis is NOT obviously holding; wait
for 5/5.

## Why we canNOT yet say "hardware edges help" (the Phase-2b lesson)
The "hardware" arm differs from flat in **three** confounded ways:
1. **Hierarchy** (DiffPool coarsening) — was null *on xla:random* (Phase 2c), untested on default;
2. **Edge-conditioned message passing** (GINEConv vs SAGEConv);
3. **Hardware contention edges** (MXU/MEM chains + edge coarsening).

On xla:random these were separately null, so the package was "safe" to test as one arm. On
xla:default the package wins — but attribution requires disaggregation.

## Required follow-up — Phase 6b disaggregation (queue AFTER sweep completes)
Run **plain DiffPool (model_phase2, no hardware edges, no GINE)** on xla:default, same 5
seeds/regime (~10–25 h GPU):
- diffpool-plain ≈ hardware (0.20) → **hierarchy** explains the win (on subtle-config data);
- diffpool-plain ≈ flat (0.03) → the **edge machinery** (GINE + contention edges) explains it;
- intermediate → both contribute; report the decomposition.
Optionally a 4th arm (GINE on data-flow edges only) to split 2 vs 3 if the edge machinery wins.

## If it survives disaggregation — what it means for the paper (do not act yet)
The story upgrades from "all nulls + noise floor" to the *stronger* pre-registered Phase-6
outcome: **architectural effects are heterogeneous across config distributions** — invisible
on xla:random (noise floor), decisive on xla:default (subtle, near-default configs where
inductive bias has room to matter), absent on nlp (easy task). Single-collection evaluation
— the field's norm — cannot see any of this. The C9 noise-floor claim is *unchanged* (it's
about sub-noise gaps; this effect is 12× the floor).

## Honest checklist before ANY paper claim
- [ ] Sweep completes (nlp:random 5/5, nlp:default 5/5) + BH correction across collections
- [ ] Phase 6b disaggregation on xla:default (which component?)
- [ ] Per-graph τ breakdown on xla:default (is flat uniformly ~0 or bimodal?)
- [ ] Rule out eval artifacts: tie structure of default runtimes (paper's 0.12 suggests fine)
- [ ] External validity: why our flat (0.033) < paper's flat (0.12) — cap-128? epochs? protocol?
