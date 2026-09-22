# Phase 8d — power addendum: why the primary test could not have confirmed P5

Fact brief (bullets only — prose belongs to the manuscript, not here).
Generated 2026-09-21 against `experiments/phase8d_results.json` (commit `4b0cbb9`).
Companion artifact: `experiments/power_analysis_mde_8d.md` (tool output, not hand-copied).

## 1. Observed paired statistics (CONV subset, n=5 seeds, paired by seed)

| comparison | Δ slowdown | paired σ (ddof=1) | Cohen's d | t-p | Wilcoxon-p |
|---|---|---|---|---|---|
| oracle-bucket vs **channel** (PRIMARY, prereg §8) | −0.0725 | **0.1029** | −0.70 | 0.1903 | 0.312 |
| oracle-bucket vs **baseline** (secondary) | −0.3768 | 0.0702 | −5.37 | 0.0003 | 0.062 |

- Per-seed primary diffs: `[+0.0253, −0.1763, −0.0153, −0.0048, −0.1913]` — sign-inconsistent
  (1 of 5 seeds runs the wrong way), which is what drives σ=0.1029.
- Arm means (CONV): baseline 1.4246 ± 0.0721 → channel 1.1203 ± 0.1077 → oracle-bucket
  1.0478 ± 0.0064.

## 2. Minimum detectable effect at n=5 (α=0.05, power=0.8, paired t)

Command: `python power_analysis.py mde --sigma 0.055 0.070 0.103 --n-seeds 3 5 7 10 15 20`

| σ | source of σ | MDE at n=5 |
|---|---|---|
| 0.055 | C9 optimistic end (project's published range) | 0.0914 |
| 0.070 | C9 worst case (project's published range) | 0.1164 |
| 0.103 | **observed** 8d primary paired σ | **0.1712** |

- Observed primary effect **0.0725 = 42% of the observed-σ detection floor (0.1712)**; it
  would need to be ≈2.4× larger to reach significance at n=5.
- Seeds required to detect Δ=0.0725 at σ=0.103: **18** (vs the 5 registered) — via the tool's
  own `required_seeds(0.0725, 0.103)`. Prereg line 114 explicitly forbids seed-adding to chase
  significance, so this was not reachable in-design.
- Internal contrast, same experiment / same n: the secondary comparison sits at 0.3768 vs its
  own floor of ≈0.1164 (3.2× above) → significant at p=0.0003. One comparison was structurally
  detectable, the other structurally was not.

## 3. P5 was unsatisfiable as written (the load-bearing point)

Inputs, all available on registration day (2026-08-19):
- Prereg lines 12–14 cite **channel CONV = 1.1203** as an established 8c result.
- Prereg line 39 sets **P5 threshold: oracle CONV ≤ 1.05**.
- Prereg line 99 sets **significance requirement: p<0.05, oracle vs channel, CONV**.
- Prereg line 114 fixes **n=5 seeds, no seed-adding to chase significance**.

Derivation:
- Maximum effect compatible with the P5 ceiling: `1.1203 − 1.05 = ` **0.0703**.
- MDE at n=5 under the project's own C9 range: **0.0914 (σ=0.055) to 0.1164 (σ=0.070)**.
- **0.0703 < 0.0914 < 0.1164** → the largest effect the threshold permits is below the
  smallest effect the significance test can resolve, even under the optimistic σ.

Implied satisfiability window for oracle CONV slowdown (must be ≤1.05 AND ≥MDE below channel):

| σ assumption | oracle would have needed | registered threshold | satisfiable? |
|---|---|---|---|
| 0.055 (optimistic) | ≤ 1.0289 | ≤ 1.05 | narrow window only |
| 0.070 (C9 worst case) | ≤ 1.0039 | ≤ 1.05 | near-empty window |
| 0.103 (observed) | ≤ 0.9491 | ≤ 1.05 | **impossible** (slowdown ≥ 1.0 by definition) |

- Observed oracle = **1.0478** — clears the ≤1.05 threshold, sits outside every detectability
  window above.
- Net: **no experimental outcome could have satisfied both halves of P5.** The threshold was
  ≈2× too loose relative to the significance requirement attached to it.
- Prereg §9's three decision buckets (confirmed / partial / floor) therefore have **no bucket**
  matching the actual outcome — "numeric threshold met, significance requirement structurally
  unreachable" was not anticipated.

## 4. Self-implication (the reason this is an asset)

- `power_analysis.py` and the P5 registration were created in the **same commit** (`0c4844a`,
  2026-08-19). The tool that shows P5 was unreachable existed when P5 was written.
- Phase 7 audit baseline for comparison: 20/20 surveyed papers report no seed variance at all
  (`phase7_meta_analysis.md` §D).
- Directional claim this supports: the underpowering trap is **structural**, not a competence
  gap — it survived authors who had built the detection tool and pre-registered in advance.

## 5. Caveats to carry into the manuscript

- σ was **not** known at registration time; §3 reconstructs it from the project's own published
  C9 range (0.04–0.07). State this as an assumption, not as a figure available on the day.
- The C9 σ range was established for **Δτ**; the 8d metric is **Δ Top-1 slowdown**. The MDE
  computation is metric-agnostic (paired-t power on σ), but the two σ scales are not
  interchangeable and should not be presented as such.
- CONV = 3 graphs × 2 collections; the n=5 pairing is over **seeds**, not graphs. Graph-level
  n is the separate, smaller limitation.
- This addendum is a post-hoc analysis of a pre-registered test. It is **not** a re-test and
  does not convert P5 into a pass.

## 6. Explicitly rejected option

- **Do not expand the frozen CONV split to reach significance.** The split is shared with 8c
  and was frozen before 8d data. Post-hoc expansion in a rigor-critique paper is
  self-defeating. A powered re-test must be a separately pre-registered extension (8e) with
  its own frozen split and its own power calculation done *before* registration.

## 7. Cross-references

- `experiments/phase8d_results.md` / `.json` — the evaluation this analyses.
- `experiments/phase8d_preregistration.md` §8/§9 — the rule being applied.
- `experiments/power_analysis_results.md` §1 — the original MDE table and C9 formalization.
- `experiments/power_analysis_mde_8d.md` — tool output for the σ values used here.
- `phase7_meta_analysis.md` §D — the audit baseline in §4.
