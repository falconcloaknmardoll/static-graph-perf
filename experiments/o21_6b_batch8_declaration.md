# O21 — Phase 6b per-graph τ at eval batch 8 (exploratory sensitivity, eval-only)

> **Status: FROZEN ON COMMIT, 2026-10-14**, before any batch-8 per-graph number exists. The first author approved
> the research-audit follow-ups in session (2026-10-14). Origin: DEFENSIBILITY O21 /
> `paper_consistency_audit_2026-10-13.md` R1.

## 0. Observation status

- **Known (batch 32, `phase6b_confound_checks.{md,json}`):**
  - inception_v3 Δ +0.568; mean of the 7 per-graph Δ 0.158; mean of the other 6 0.090.
  - BH (flat-vs-hier / hier-vs-hw / flat-vs-hw) .00285 / .0299 / .000149.
  - Tightening (flat): median SD 0.0152 at n = 7 (0.0071–0.0549), 0.0135 at n = 20.
- **Already known at batch 8:** each checkpoint's mean over the 7 graphs is its registered val τ.
  Hardware 0.193 / 0.2118 / 0.1959 / 0.1788 / 0.209 and flat 0.0269 / 0.031 / 0.0213 / 0.0285 / 0.0561
  (`phase6e_results.json`). So the per-seed means are known; the per-graph split is not.
- **Never computed:** per-graph τ at batch 8.
- **Code check:** the analysis code was dry-run on the batch-32 JSON only. It reproduces every number
  above exactly. No inference was run.

## 1. Design (fixed now)

- **Checkpoints:** `checkpoints/expansion/xla_default/{wide-baseline,hardware}-seed<s>`,
  s ∈ {42, 100, 123, 456, 789}.
- **Code:** `phase6b_confound_checks.per_graph_tau`, unchanged, with `eval_batch_size = 8`.
- **Script:** `scripts/phase6b_batch8_rescore.py` → `experiments/phase6b_confound_checks_b8.{md,json}`.
  The batch-32 files are never rewritten.
- **Gate:** each seed's 7-graph mean must equal the stored batch-8 val τ. If it doesn't, the
  difference is reported and the batch-8 rows are flagged.
- **Reported at both batches:**
  - the per-graph Δ table
  - inception_v3 Δ, the mean of all 7, and the mean of the other 6
  - the 3-way BH, with flat-vs-hardware at the batch in question
  - the tightening curve at n = 7 and n = 20 (same bootstrap seed and grid as the original)

## 2. How it will be used (fixed now)

The numbers are appendix material: the lr-1e-3 xla:default contrast, which option A no longer leads with.
- **Which numbers the paper uses:** batch 8, the paper's protocol, with batch 32 beside them.
- **If inception_v3's share changes materially** (its Δ more than doubles or halves relative to the
  others), the O3 sentence quotes the batch-8 share and notes the change.
- **FIG-M2:** quotes whichever batch matches the methods sentence (batch 8), with the batch-32 value
  in its caption if they differ in the third decimal.
- **No verdict depends on this.** No extra seeds, no other batches.

## 3. Budget and order

Eval-only: 10 checkpoints × 7 graphs, about 10–20 GPU-min. Queue `scripts/run_gpu_queue_2026-10-14p.sh`
waits for queues k, l, m, n and o and the 8e evaluation window. **Cutoff:** not started after
2026-10-23 23:59 ET.

## Sign-off

- [x] the first author approval: 2026-10-14 (given in session; recorded by Claude)
- [x] Frozen on commit: the commit that adds this file
