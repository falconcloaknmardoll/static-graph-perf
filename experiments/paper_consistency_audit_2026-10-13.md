# Research consistency audit (2026-10-13/14)

Read-only. Nothing was trained, launched or edited. Computations are arithmetic on existing files
plus one CPU input-equality check.

Scope: the research itself:
- data splits
- evaluator code
- the saved args of all 161 checkpoints
- every evaluation script that produced a reported number
- results JSONs, recomputed from their per-seed values
- DEFENSIBILITY / VERIFICATION / outline / Desk status

The Google Doc draft is out of scope (parked at the end).

**Bottom line:** no finding changes a registered verdict. Four are worth acting on before Oct 23:
- R1: more undisclosed batch-32 evaluations.
- R2: the two "flat at lr 1e-3" runs differ by 0.057.
- R3: the 8d oracle checkpoints are all from epoch 5.
- R4: 8e is ready to evaluate.

## Findings

### R1. Four more analyses scored at eval batch 32, undisclosed (same class as O8/O19)

| Analysis | Script (default batch) | Feeds | Exposure |
|---|---|---|---|
| Phase 5 per-graph Top-1 (1.44–1.51× conv vs 1.01–1.07×) | `phase5_error_analysis.py:125` (32) | §6 ¶1, FIG-C1, the §6 ¶4 tables in the Desk; C10 | Flat + hardware xla:random checkpoints. These are the arms whose τ moves ≈ −0.003 / −0.016 at batch 8. A Top-1 shift of that size cannot erase a 1.44 vs 1.07 split, but the paper will not say which batch was used. |
| Phase 6b per-graph τ (inception_v3 Δ +0.568 ≈ half the effect; O3) | `scripts/phase6b_confound_checks.py:39` (32) | §5 inception_v3 disclosure, appendix | Hardware per-seed means here are 0.1755 / 0.2232 / 0.1749 / 0.1792 / 0.2037. The registered batch-8 values are 0.193 / 0.2118 / 0.1959 / 0.1788 / 0.209. Flat matches to ≤ 0.0003. |
| Tightening curve (FIG-M2: median SD 0.0152 at n = 7) | `power_analysis.py tighten` on `phase6b_confound_checks.json` key `per_graph_tau.wide-baseline` | §4 ¶4 | Flat arm only. The batch effect is negligible (≤ 0.0003 per seed), so this needs a disclosure only. |
| Phase 4 Top-K on xla:random (p = .47–.65) | `phase4_evaluate_topk.py:93` (32) | Desk §5 ¶3 cites it | Already superseded by `topk_slowdown_xla_random.json` (O9, batch 8, all p ≥ .45). **Cite the batch-8 file instead.** |

Also batch 32, but consistent with its own declaration: the 8c leave-one-graph-out check (reads 8c's
batch-32 results, as 8c does) and the 8c collapse diagnostic (exploratory).

Batch 8 as recorded: `topk_slowdown.py` (`--eval-batch` default 8), 6d/6e/6f (training-time val τ),
both eval-batch-sensitivity files.

**Suggested handling:** add these rows to the eval-batch block (outline §4, Desk §4 ¶5) and to the
appendix table as "scored at batch 32".
- **Phase 5:** R0 will produce the same checkpoints' per-graph CONV/OTHER Top-1 at batch 8 as a declared
  sensitivity. After R0 is evaluated, those cells can be quoted beside Phase 5 at no extra GPU cost.
  **Do not read R0's existing caches before then.** R0's declaration keeps its numbers unseen.
- **6b:** a batch-8 re-run of the per-graph τ is eval-only (≈ minutes per arm; it would need a
  one-line declaration). Alternatively, disclose it as batch 32, since it is appendix material on the
  lr-1e-3 contrast that option A no longer leads with.

### R2. The two flat-at-lr-1e-3 runs on xla:random differ by 0.057 τ

- **The runs:** §5's flat model is the Phase 2c run (val τ 0.288 at batch 8). §7's 8c baseline is
  meant to be the same model: same class, LR, data, seeds, and selection at batch 8. Its mean is
  **0.231**.
- **Per seed:** 8c − 2c = −0.080, +0.036, −0.078, −0.058, −0.105. Mean −0.057, paired p = .08.
- **Inputs are identical:** `TPUGraphDataset` and `MXUGraphDataset(baseline)` give byte-identical
  tensors (checked on 3 validation samples).
- **What differs:** the training harness. `num_workers` 4 vs 0 changes the random config sub-sampling
  stream, and `train_8c.py` has its own copy of the training loop.
- **Status:** noted, not explained, in `framing_audit_2026-10-11.md:87-93`.

**Why it matters:**
- Every paired contrast is within one harness, so no verdict is affected.
- But the same recipe, re-run, moved the mean by 0.057. That is close to the n = 5 MDE (0.066), and
  one seed flipped sign. It fits the paper's thesis (C9: run-level randomness rivals the effects).
- It also means:
  - §5's "flat" and §7's "baseline" must never be quoted as one number.
  - The outline's "fair contrast" (hier −0.0235 vs channel −0.3043) compares contrasts from two
    harnesses, and should say so.
  - Channel's +0.10 overall-τ gain over its own baseline would look like +0.045 against the Phase 2c
    flat model. Never mix the two.

**Suggested handling:** one sentence in §7 (the baseline's own τ, and that a separate run of the same
recipe scored 0.288). Add an objection to DEFENSIBILITY.

### R3. Every 8d oracle checkpoint is from epoch 5 (undisclosed)

All 5 seeds peaked at the first validation (val τ 0.79–0.80) and drifted to about 0.77 by epoch 50
(`logs/capped/xla_random_oracle-bucket_seed42.log`; `best_model.pt` epoch = 5 × 5).
- **Not a protocol breach:** the selection rule is the same as for every arm.
- **Why disclose:** a reviewer could read it as "oracle trained 5 epochs vs 50".
- **Suggested handling:** one clause in §7 ¶5.

### R4. 8e is ready to evaluate, and the Desk still says "running"

- **Done:** queue k finished 10-12 22:03, and the boundary steps for both arms finished 10-13 18:03
  (`logs/gpu_queue_2026-10-11l.log`).
- **LR\*:** 3e-5 for both arms (the R4 launch line).
- **Missing:** there is no `phase8e_results.*` yet.
- **Next step:** Q1/Q2 primary (batch 32) can be evaluated now. Its batch-8 sensitivity cells for
  the lr-1e-3 8c arms will come from O19 (~Oct 15).
- **Stale Desk slot rows:** 8e "running" and R4 "queued" should be updated.

## Checked and consistent

- **Data:**
  - No graph appears in both train and valid, within any collection or across collections. This
    matters because 8c scores xla:random-trained models on xla:default/valid, and there is no
    overlap there either.
  - The two xla valid sets are the same 7 graphs, so 8c's 6 CONV instances are 3 graphs × 2
    collections. This is already disclosed (O20).
  - Valid counts: 7 for xla, 20 for nlp.
- **τ code:** `train.evaluate`, `eval_fast` and the 6b script all use scipy τ-b per graph and average
  over graphs. NaN graphs are dropped. The only run with dropped graphs is the nlp:default collapse
  (`val_n_graphs = 0`), and the 6d pre-registration (:222-225) already describes it correctly as
  undefined τ, not zero.
- **Saved args:** the only differences across groups are lr (intended), eval_batch (O8),
  num_workers (R2), data_root (intended) and fast_eval. fast_eval is mixed within the nlp:default
  flat arm, and that is documented with an equivalence check (`phase6d_…:296-309`).
- **Results JSONs:** 6e, 6f, 6d-amended and Top-K (xla:default, nlp:random) all match their reported
  Δ and p. 8c, P4 and 8d reproduce exactly from the caches (checked earlier today).
- **Parameter counts:** 236,157 / 235,553 / 234,120 (`run.log`). Flat vs hardware is 0.86% apart,
  not 0.3%; the Desk says "within 0.3%" for the hardware arm (§4 ¶1 Evidence, §5 hardware Address).
- **MDE:** the tool gives 0.06650 exactly, so 0.066. `power_analysis_results.md:15,20` and
  `paper/sections/04_instrument.tex` still say 0.067.
- **Erratum `9648419`** (predominantly compute-bound; ρ +0.74) is reflected in the Desk.
- **VERIFICATION:**
  - C1 (0.34) and C5 (63% compute) are 🔴, and neither is cited in the outline or the Desk.
  - C11 (ridge ≈ 137) is 🟡 and is used in §6 ¶3. It needs a human check.

## Stale scaffolding (mechanical)

- `paper/sections/05_result_distribution.tex` P1–P3: "clear win", p = .22 / .72, old figure files.
- `paper/sections/06_result_conv_gap.tex` P2: n = 160 (now 95/49).
- Desk writing-rules example: "n=160 graphs".

## Not covered

- Prereg-by-prereg rule compliance beyond the headline verdicts (audited when each was evaluated).
- Citation claims.
- §2–3 facts.
- Gate D (Oct 17–18) traces every number.

## Parked: Google Doc draft notes (for when the first author revises; not part of this audit)

1. §8 ¶2 names TPUGraphs for the A2 correction; it was TGraph.
2. §8 ¶4 quotes the batch-32 DiffPool contrast (−0.0117, p = .22), while §5 ¶2 quotes batch 8.
3. §4 ¶7 says the shared LR is 1e-4; it is 1e-3.
4. §6 ¶3 says every conv is compute-bound; see erratum `9648419`.
5. The intro says 90%; it is 85%.
6. The intro, contribution 1 and the §5 heading predate option A.
7. Smaller:
   - "within 0.3%" (should be 0.9%).
   - "width 172" is the hidden size.
   - nlp has 20 val graphs.
   - The τ SD and the Δτ SD are conflated.
   - The edge windows are schedule-order windows.
