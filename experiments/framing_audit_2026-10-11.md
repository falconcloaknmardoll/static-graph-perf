# Framing audit — 2026-10-11 (read-only)

Scope: Option A framing ("structure vs information") and the proposal to lead with the
convolution/MXU result. No files edited except this one; no jobs launched; no pre-registration
touched. Every number cites its source as `file:line`. Where I recomputed something from repo data,
it is marked **[recomputed, post-hoc]** with the method. Pre-registered vs post-hoc is marked on
every claim. One limit on coverage: of the Manuscript Desk I read only the Paragraph-skeleton tab
(§6, §7 and the option-A lines). The "Manuscript Desk" and "Primer" tabs were not checked.

---

## Bottom line

1. **8c's baseline is the same flat model, at the same untuned lr 1e-3, as the arm 6e showed to be
   handicapped.** But it trained on xla:random, and it does not show 6e's failure signature. Verdict on
   whether part of 1.42 → 1.12 is a baseline-LR artifact: **can't tell.** Nothing in the repo rules it
   out, and three facts make it a live objection (§1d). Because the paper's own §5 argument is exactly
   "an untuned lr-1e-3 baseline can manufacture a win", a reviewer will apply it to §7.
2. **The model-free convolution claim (§6) is the strongest thing in the paper.** The reported
   n's are wrong: 16 test graphs with no runtimes are counted. **Headline with the counting fixed:
   n = 95 / 49, ratio 2.77×, Mann-Whitney p = 1.5e-7 [recomputed].** The de-duplicated 4.35× and the
   per-collection replications are post-hoc robustness checks, not a replacement headline (see the
   correction at the end).
3. **Option A's contrast, as currently written, compares an LR-tuned null (xla:default) with an
   untuned positive (8c, xla:random).** A fair like-for-like version exists inside the data (§3a):
   on xla:random, at one shared lr 1e-3, structure gives no gain but MXU information does.
4. **Recommendation: one pre-registered run block before Oct 23** (an 8c baseline+channel LR check,
   about 60 GPU-h), plus one free eval-only check. Don't make convolution the lead unless the LR check
   comes back clean. Details in §4.

---

## Task 1 — LR-handicap check on 8c

### 1a. What 8c trained and evaluated on

- **Training collection:** `layout:xla:random` train split (`phase8c_preregistration.md:100`,
  Amendment A1.1). The launcher uses `COLL="xla/random"` (`run_phase8c.sh:23`; scramble
  `run_phase8c_scramble.sh:22`). The baseline seed-42 log shows `train : 69 graphs, 1,005,926 pairs
  in_dim=158`.
- **Checkpoint selection:** best overall val τ on `xla/random/valid` (all 7 graphs). `train_8c.py:58`
  loads `<data_root>/valid` with data_root = xla/random; `train_8c.py:106–110` saves on best τ.
- **Evaluation (primary metric):** CONV Top-1 slowdown on the frozen 3-graph CONV subset
  (inception_v3_batch_128_train, resnet50.4x4.fp16, resnet_v1_50_official_batch_128_bf16;
  `phase8c_preregistration.md:45–46`), scored on **both** `xla/default/valid` and `xla/random/valid`,
  giving 6 instances (`phase8c_preregistration.md:50–51`). **So yes, the CONV set includes the
  xla:default copies of the 3 graphs,** but only as evaluation targets for a model trained on
  xla:random.

### 1b. Recipe and model

- **The model is the flat GraphSAGE:** `BaselineGraphSAGE_Large` ("wide-baseline", 236,157 params;
  `train_8c.py:78`; `phase8c_preregistration.md:108`). This is the same model class as 6e's flat arm
  (`phase6e_flat_lr_preregistration.md:32`, "flat (`wide-baseline`, h=172, ~236k params)").
- **LR:** lr 1e-3 cosine, 50 epochs, wd 1e-4, grad-clip 0.1 (`launch_capped.sh:122`; the
  checkpoint args for `checkpoints/phase8c/xla_random/mxu-baseline-seed42/best_model.pt` read lr 0.001,
  epochs 50). This is the same registered recipe that 6e found handicaps flat **on xla:default**
  (flat 0.0327 at 1e-3 vs 0.2255 at LR\* = 1e-4; `phase6e_results.md:18,20`).
- **Every 8c arm uses this one LR.** Nothing in 8c was LR-tuned (`phase8c_preregistration.md:65–69`).

### 1c. Evidence on undertraining

Training logs `logs/capped/xla_random_mxu-<arm>_seed<s>.log` (the `Done. Best val_τ` line, plus
epoch-1 and epoch-50 loss):

| arm (device) | best val τ (seeds 42/100/123/456/789) | mean **[recomputed]** | best epoch | loss ep1 → ep50 |
|---|---|---|---|---|
| baseline (GPU) | .2367 .2715 .2279 .2063 .2131 | 0.2311 | 50, 45, 50, 50, 50 | 1.27–1.36 → 1.08–1.12 |
| channel (GPU) | .3657 .3487 .3249 .3838 .2397 | 0.3326 | 50 ×5 | 1.26–1.37 → 1.07–1.12 |
| scramble (CPU) | .2541 .2165 .2459 .2205 .2742 | 0.2422 | 50, 30, 50, 50, 50 | 1.25–1.40 → 1.06–1.11 |

Compared with 6e:

- **The 6e failure signature is absent.** 6e's flat@1e-3 on xla:default barely moved its loss
  (1.304 → 1.291; `phase6e_flat_lr_preregistration.md:44`) and reached τ 0.0327. The 8c baseline's
  loss falls about as much as the channel arm's, and it reaches τ 0.21–0.27.
- **But it is not converged either.** 4/5 baseline runs (and 5/5 channel runs) peak at the last epoch,
  the same "within a 50-epoch budget" condition that 6e/6f had to disclose
  (`phase6e_results.md:49–50`).
- **The baseline ranks convolution configs in the wrong direction.** The collapse diagnostic
  (`experiments/phase8c_collapse_diagnostic.json`, `scripts/phase8c_collapse_diagnostic.py`) gives a
  mean Spearman(pred, runtime) on CONV of **+0.048 (xla:default) and +0.120 (xla:random)** for the
  baseline. The sign is wrong: Top-1 = argmax, so a good model has ρ ≪ 0. On OTHER/xla:random it is
  −0.645. Channel flips CONV to −0.118 / −0.237 **[recomputed means from the JSON]**. The timeline
  already records this diagnostic as "MIXED", clear on xla/random and weak on xla/default
  (`PROJECT_TIMELINE.md:73–77`).
- **The number the Desk uses to say "flat trains normally on xla:random" is not the 8c baseline's.**
  The 0.288 (`paper_outline.md:52,60`; Desk Revision log) is the Phase 2c flat arm
  (`eval_batch_sensitivity_xla_random.md:7`, +0.2880 ± 0.0365). The 8c baseline, same recipe but a
  different harness (num_workers 0 vs 4, per A1.5 and the checkpoint args), averages **0.2311**. Same
  seed 42: Phase 2c checkpoint val τ 0.3193 (selected at eval batch 32) vs 8c 0.2367 (batch 8)
  **[read from checkpoint dicts]**. 0.2311 is still above the published 0.19, so "trains" holds. But
  the sentence should cite the 8c baseline's own number.

### 1d. Verdict: **can't tell**

- **Against an artifact:** the 6e mechanism (loss stuck, τ ≈ 0.03) is absent. The scramble arm, at
  the same LR and capacity, does **not** reach channel's level (`phase8c_control_results.md:29`). So
  whatever the baseline lacks, uninformative extra columns don't supply it. The effect also appears
  in **all 6** CONV instances (table in Task 2), not one outlier.
- **For a possible artifact:**
  1. Same model, same untuned lr 1e-3, under-converged at 50 epochs.
  2. MXU features are config-invariant, so the effect can only act through **training dynamics**
     (`phase8c_preregistration.md:119–122`, A2 reason). Training dynamics is exactly what an LR
     change alters.
  3. The baseline's CONV rankings are inverted (ρ > 0). That is a learnable failure, and a better-tuned
     baseline could plausibly reduce it.

  Channel also raises **overall** xla:random val τ by about +0.10 (0.3326 vs 0.2311), which looks more
  like a global optimization change than a CONV-local one, even though OTHER Top-1 is unchanged.
- **What would settle it:** an 8c baseline LR sweep (§4, R1). Until then, the paper can't claim that 8c
  is immune to the failure mode it reports in §5. The Desk's "Level 2 is NOT exposed to the LR
  artifact" (`paper_outline.md:58–62`) is too strong. Accurate: "8c does not show 6e's failure
  signature; its baseline was not LR-tuned."

---

## Task 2 — Robustness of the convolution claims

Per-instance CONV Top-1 means **[recomputed from `phase8c_results.json` raw, 5 seeds each]**:

| instance | baseline | channel | spatial | full | scramble |
|---|---|---|---|---|---|
| default · inception_v3 | 1.4534 | 1.1034 | 1.0214 | 1.0890 | 1.3724 |
| default · resnet50.4x4 | 1.2983 | 1.0682 | 1.0495 | 1.1337 | 1.0388 |
| default · resnet_v1_50 | 1.0433 | 1.0070 | 1.0084 | 1.0070 | 1.0417 |
| random · inception_v3 | 1.6154 | 1.2395 | 1.2704 | 1.3917 | 1.6520 |
| random · resnet50.4x4 | 1.6694 | 1.1122 | 1.4977 | 1.2439 | 1.5151 |
| random · resnet_v1_50 | 1.4681 | 1.1914 | 1.2075 | 1.2945 | 1.5671 |

Channel beats baseline on 6/6 instances, but resnet_v1_50/default has almost no headroom
(1.0433). The CONV mean is driven by 5 instances from 3 graphs.

### Claim-by-claim

| Claim | Exact numbers (source) | n | Weakest point | Can a hostile reviewer knock it over? |
|---|---|---|---|---|
| **Conv graphs 2.8× more layout-sensitive** (model-free) | median runtime CV 0.239 vs 0.086 (`phase8_mxu_misalignment.md:29–30`) | reported 105 / 55; **actual 95 / 49** | **n's are miscounted** (see mismatches below); no test statistic reported; graph names counted twice across collections | **No, once fixed.** Headline: [recomputed] finite-runtime graphs 95 vs 49, ratio 2.77, MWU p = 1.5e-7. Post-hoc robustness checks, labelled as such: unique graphs (averaged across collections) 49 vs 27, medians 0.268 vs 0.062, 4.35×, MWU p = 1.8e-5; per collection default 3.68× (p = 7.4e-4), random 3.34× (p = 2.8e-5). |
| **MXU fraction vs runtime CV ρ = +0.54** | `phase8_mxu_misalignment.md:31` | 144 | The script's Spearman uses argsort ranks with **no tie averaging** (`scripts/analyze_mxu_misalignment.py:103`); mxu_frac has 49 tied zeros | Minor. [recomputed] tie-corrected ρ = **+0.52** (p = 2.5e-11); unique graphs ρ = +0.55 (n = 76, p = 2.9e-7). Replace +0.54 with the tie-corrected figure. |
| **Three static explanations fail** | (a) overhead vs CV ρ = −0.06, n = 95 (`:32`); (b) roofline: UNet-3D AI ≈ 2,857 at 1.01×, ResNets AI ≈ 700 (`:60–62`); (c) T2 order mismatch, n = 3 (`:20–24`) | 95 / 4 graphs / 3 | (b) and (c) are n = 4 and n = 3 descriptive orderings. Only (a) is a test. | (a) holds ([recomputed] scipy ρ = −0.065, p = .53). That is a null, not evidence of no relation; say "no detectable correlation". (b) and (c) are anecdotes and should be labelled descriptive. |
| **Channel arm 1.4246 → 1.1203** | Δ −0.3043, CI [−0.3713, −0.2271], t-p .0021, **BH-p .0032**, Wilcoxon .062, d −3.15 (`phase8c_results.md:6–7,12`) | 5 seeds × 6 instances (3 graphs) | **3 graphs**; untuned LR (Task 1); indirect mechanism; P1 reversed | Statistically solid at n = 5. [recomputed] paired diffs −.3186 −.1512 −.3700 −.3990 −.2830, sd 0.0966, MDE at n = 5 = 0.1606, effect is 1.9× MDE. **Knockable on generalization (3 graphs) and on the LR question.** Not knockable on noise. |
| **Scramble control (P4)** | channel − scramble Δ −0.2442, CI [−0.3154, −0.1606], t-p .0063, Wilcoxon .062, d −2.35 (`phase8c_control_results.md:29`) | 5 | CPU (scramble) vs GPU (channel), O17 | P4 itself holds: Δ = 0.2442 vs MDE 0.1731 at σ 0.1041 [tool]. **But "scramble ≈ baseline" (Δ −0.0601, p = .30) is not evidence of equivalence:** MDE for that contrast is 0.1882 at σ 0.1132, and 30 seeds would be needed to detect 0.0601 [tool]. DEFENSIBILITY O17 (`DEFENSIBILITY.md:67`) uses that p = .30 as "No evidence of bias". That reasoning is weak and should be dropped or rephrased. |
| **Oracle 1.0478; 81% recovery; P5 not confirmed** | oracle 1.0478 ± 0.0064; vs channel Δ −0.0725, CI [−0.1501, +0.0052], t-p .1903, Wilcoxon .312, d −0.70; recovery 0.8077 (`phase8d_results.md:9,12,26`) | 5 | Oracle trained on CPU (`logs/capped/phase8d_master.log:12`); recovery fraction is a ratio of noisy means | P5 not confirmed, as reported. 18 seeds needed (`phase8d_power_addendum.md:31`; tool agrees). 81% must stay descriptive. Oracle runs peak at epoch 5 (log `Done` lines: .7923–.8019), as expected for a leaky signal. |

### The specific weak points the brief asked about

- **3-graph CONV subset.** The biggest generalization weakness. Pairing is over seeds, not graphs
  (`phase8d_power_addendum.md:84–85`). The frozen split is the benchmark's maximum held-out conv
  graphs (`phase8c_preregistration.md:51–52`). That is true for *validation*, but xla train has 46–49
  conv-bearing graphs [recomputed], so a held-out-train design is possible (§4, R4, exploratory).
- **Wilcoxon .062.** This is the n = 5 two-sided floor (all 5 seeds same sign), so it is as low as the
  test can go, not a weakness. Say "all five seeds in the same direction".
- **Channel's variance (± 0.1077).** Seeds 100 (1.2298) and 789 (1.2432) are much worse than the others
  (1.016–1.067) (`phase8c_results.json` per_arm_seed_means). This drives 8d's sign-inconsistent primary
  test (`phase8d_power_addendum.md:14–15`). It is honest variance, but it means "1.12" hides a
  bimodal 1.02 / 1.24 pattern. Consider showing the per-seed values.
- **Config-invariant features and the indirect mechanism.** The weakest scientific point for a hardware
  audience. Features can't distinguish configs (`phase8c_results.md:45–51`). The proposed mechanism
  ("informative prior", `:83–90`) is **post-hoc** and only partly supported by the collapse diagnostic
  (clear on xla/random, weak on xla/default; `PROJECT_TIMELINE.md:73–77`). A hardware reviewer will
  ask how a feature identical across all layouts helps pick a layout. The honest answer is "it changes
  what the encoder learns", which is also what an LR change does (Task 1).
- **Reversed P1.** Registered null for channel, observed the strongest effect
  (`phase8c_preregistration.md:33`; `phase8c_results.md:36`). Report it plainly; the Desk already says
  so. Note that the between- vs within-graph reconciliation (`phase8c_results.md:62–77`) is
  **post-hoc**.
- **OTHER-placebo wobbles.** Full −0.0147, CI [−0.0215, −0.0079], p = .0199 (`phase8c_results.md:19`).
  Scramble +0.0079, CI [+0.0038, +0.0121], p = .0283 (`phase8c_control_results.md:16`). Both are tiny,
  and under §8 the full arm's wobble triggers "investigate" (`phase8c_preregistration.md:86`). New
  point: **the scramble wobble is the only placebo comparing a CPU-trained arm with a GPU-trained
  baseline**, so O17 is a candidate explanation. Can't tell from the repo.
- **CPU vs GPU (O17).** Scramble and oracle on CPU; baseline and channel on GPU
  (`DEFENSIBILITY.md:67`, verified in log `Device` lines). It affects P4 and 8d's primary test. See the
  scramble ≈ baseline caveat above.

### Number mismatches found (source vs where it's quoted)

1. **n = 105 / 55 / 160 are wrong.** The T3 scan globs `xla/*/*/*.npz`, which includes `test/`
   (`scripts/analyze_mxu_misalignment.py:160`), where `config_runtime` is all zeros (DEFENSIBILITY
   2026-10-10 Kaggle log). Those 16 graphs have NaN CV, are excluded from the medians, but are counted
   in the printed n. [recomputed] finite: **95 conv / 49 non-conv / 144 total**. Medians 0.239 / 0.086
   are unchanged. Appears in: `phase8_mxu_misalignment.md:27–30`, `paper_outline.md:291–292`,
   `phase8c_preregistration.md:14` ("160 xla graphs"), Desk §6 ¶2 ("n = 160 … (n = 105) … (n = 55)").
2. **ρ = +0.54** has no tie handling. Tie-corrected +0.52 (above).
3. **Channel BH-p.** The registered BH across the 3 feature arms is **.0032**
   (`phase8c_results.md:12`). The control file recomputes BH over {channel, scramble} → .0043
   (`phase8c_control_results.md:11`). The outline quotes **.004** (`paper_outline.md:114`) and the
   novelty draft says "BH-p=.004". Use .0032; the Desk already does.
4. **`phase8d_power_addendum.md:72–73`** says "20/20 surveyed papers report no seed variance at all".
   The audit records Axis 1 as **16 F / 4 ∂** (`phase7_meta_analysis.md:565`). Stale. Don't let it reach
   the paper.
5. **Desk §7 ¶7** still says "the 81% recovery favours representation", in the same bullet as the O16
   decision that forbids that reading. Contradiction within one bullet.
6. **Desk §7 ¶1 wording:** "MXU tiling geometry" describes the spatial arm better. The channel arm's
   columns are c_out/K MXU utilisation and padding waste (`phase8c_preregistration.md:104–105`).

All other checked numbers match their sources: 1.4246, 1.1203, −0.3043, d −3.15, −0.2442, .0063,
−0.0601, .3008, 1.0478, 0.8077, −0.0725, .1903, the placebo values, ρ = −0.06 / n = 95, and the
Phase 5 slowdowns 1.441 / 1.452 / 1.512.

---

## Task 3 — Framing check

### 3a. Does Option A hold up? Is the contrast fair?

**As currently worded, partly unfair.** The "structure" side's strongest null is LR-tuned (6f,
xla:default). The "information" side's positive is untuned (8c, lr 1e-3, xla:random). That compares
an LR-tuned null with an un-tuned positive, which is the asymmetry the brief asks about.

**A fair version already exists in the data.** On **xla:random, at the single shared lr 1e-3, same
flat backbone**:

- Structure vs flat: hierarchy −0.0235 (p = .070), hardware −0.0241 (p = .464) τ at b = 8
  (`eval_batch_sensitivity_xla_random.md`). Top-K null, p = .47–.65 (Desk §5 ¶3,
  `phase4_topk_results.md`).
- Information vs flat: channel Δ CONV Top-1 −0.3043, BH-p .0032.

That is like-for-like in collection and recipe. It is **not** like-for-like in metric (overall τ /
overall Top-K vs CONV-subset Top-1) or in harness (Phase 2c ran with 4 workers, 8c with 0). The
eval-only check R0 (§4) would make the metric match at zero training cost.

Option A survives if it's stated as: **"at one shared recipe on xla:random, adding structure gives
no detectable gain while adding MXU information narrows the conv gap. On xla:default, the one
apparent structural win dissolved under LR tuning, and the MXU result has not been LR-tuned."** The
last clause must stay until R1 runs.

### 3b. If convolution became the lead

Load-bearing claims, strongest first:

1. **Model-free sensitivity gap.** This should be the lead's foundation: 2.77× (n = 95 / 49, MWU
   p = 1.5e-7), holding in post-hoc robustness checks (unique graphs; each collection), no model involved. It carries its weight once the n's
   are fixed.
2. **8c channel effect + P4.** Statistically clear at n = 5 and present on 6/6 instances. **Not yet
   strong enough to lead**, for three reasons: (i) 3 graphs; (ii) the untuned-LR question the paper
   itself raises; (iii) a config-invariant, post-hoc mechanism that a hardware committee will press
   on. (ii) can be fixed before Oct 23. (i) can be partly addressed (exploratory). (iii) can only be
   disclosed.
3. **8d.** Descriptive only; can't carry weight.
4. **Three static negatives.** Only (a) is a test; (b) and (c) are illustrations.

**Verdict:** a convolution-led paper is defensible **only if R1 comes back clean** (channel still beats
an LR-tuned baseline). If R1 shows the gap shrinking a lot under tuning, the conv result has the same
fate as Finding 1, and leading with it would be the worst option.

### 3c. Framing inconsistencies across repo docs

| Location | Stale or incorrect text | Issue |
|---|---|---|
| `paper/sections/05_result_distribution.tex:1` | `\section{Result I: Graph Information Is Distribution-Dependent}` | pre-option-A title |
| `experiments/paper_outline.md:181` | "### 5. Result I — graph information is distribution-dependent" | same |
| `experiments/paper_outline.md:27–28` | thesis line "Distribution-dependent … decisive+decomposable" | marked superseded but still the anchor text |
| `experiments/paper_outline.md:41` | "on every collection with results" | unscoped "every" (option A description, plan A) |
| `experiments/paper_outline.md:58–62` | "Level 2 is NOT exposed to the LR artifact … τ 0.288" | too strong (Task 1); 0.288 is the Phase 2c flat, not the 8c baseline (0.2311) |
| `experiments/paper_outline.md:112–114` | "Finding 3 … CONFIRMED … significantly closes the CONV Top-1 slowdown gap … BH-p=.004" | "closes" (Desk forbids it); BH-p should be .0032 |
| `experiments/paper_outline.md:291–293` | n = 160, 105, 55; ρ = +0.54 | miscounted n; untied ρ |
| `experiments/paper_outline.md:238` | "+0.1650 … p<0.001 … ~12×" | fine as a fact bullet, but must carry "at shared lr 1e-3" |
| `PROJECT_TIMELINE.md:336` | "Level 1 — graph info is distribution-dependent … framing open" | stale; option A decided 2026-10-10 |
| `PROJECT_TIMELINE.md:711` | "'distribution-dependent' upgrades to a pattern" | stale plan text |
| `experiments/novelty_venue_draft.md` §1–2 | "L1 — xla:default split reveals a real, decomposed architectural signal"; "split by evaluation collection"; "close the conv ranking gap"; "BH-p=.004"; 8d "prove the floor is representational" | whole doc predates 6e/6f, 8d and option A, and is still marked "FINALIZED" |
| `experiments/phase8c_results.md:96–98` | "MXU geometry features close the convolution ranking gap" | dated interpretation; contradicts line 99's own "NOT … fixed" |
| `experiments/phase8_mxu_misalignment.md:65–66` | "sharpens the representational-floor thesis … no static shape-derived feature captures" | superseded by 8c's positive; written before 8c |
| `experiments/phase8d_power_addendum.md:72–73` | "20/20 surveyed papers report no seed variance at all" | audit says 16 F / 4 ∂ |
| `DEFENSIBILITY.md:67` (O17) | "No evidence of bias: scramble (CPU) ≈ baseline (GPU), Δ −0.0601, p = .30" | underpowered null used as evidence of absence (MDE 0.188) |
| Desk §6 ¶2 | "n = 160 … (n = 105) … (n = 55); ρ = +0.54" | as above |
| Desk §7 ¶1 | "MXU tiling geometry" | channel = padding utilisation |
| Desk §7 ¶7 | "the 81% recovery favours representation" | contradicts O16 in the same bullet |
| Desk Revision log (option A item) | "Level 2 is not exposed … τ 0.288 > published 0.19" | as `paper_outline.md:58–62` |

No hits for "residual is representational" as a claim (only as a prohibition), nor for "fixes
convolution" outside prohibitions. The `.tex` files are skeletons (143 lines total), and only the §5
title is stale.

---

## Task 4 — Costed run plan (proposal only; nothing launched)

**Wall-clock basis:** 8c GPU runs took 9,764–12,866 s each, about 2.7–3.6 h
(`logs/capped/phase8c_master.log` DONE lines). Epochs run about 100–113 s plus evaluation. CPU runs
took about 10.3–10.8 h (`PROJECT_TIMELINE.md:89–91`). The GPU is idle as of this audit (4 % util):
queue j finished Oct 11 00:16 (`logs/gpu_queue_2026-10-05j.log`). The machine is shared, so estimates
assume about 3 h/run with no contention. **Budget to Oct 23 23:59 ≈ 12.5 days.**

Ranked by robustness gained ÷ cost:

| # | Run | Tests / answers | Arms × seeds | Est. wall-clock | Fits by Oct 23? | Status needed |
|---|---|---|---|---|---|---|
| **R0** | Eval-only: CONV-subset Top-1 for the existing Phase 2c/3 xla:random checkpoints (flat, hierarchical, hardware) | Makes the option-A contrast like-for-like in metric (§3a) | 0 training; 15 checkpoints | ~2–4 h CPU/GPU eval (estimate; not timed) | yes | **exploratory**; declare before computing (as `6def25d`) |
| **R1** | **8e: LR check on 8c**, 6e/6f-style. Baseline and channel at lr {1e-4, 3e-4} (1e-3 exists), boundary step at 3e-5 if LR\* hits the grid edge. LR\* chosen by **overall xla:random val τ** (the existing selection metric), **not** CONV Top-1. Primary = channel@LR\*_ch − baseline@LR\*_b on CONV Top-1. | Task 1: is 1.42 → 1.12 partly a baseline-LR artifact? Pre-empts the paper's own §5 objection against §7 | 2 arms × 2 LRs × 5 = **20 runs** (+10 if the boundary step fires for both) | **≈ 60 h** (+30 h) ≈ 2.5–3.75 days | **yes**, if launched by ~Oct 15 and the GPU is not taken | **new pre-registration required** (frozen before launch; the 8c prereg forbids new arms after data, `phase8c_preregistration.md:91`, so it must be a separate registered extension, as `phase8d_power_addendum.md:91–94` already recommends) |
| R4 | Held-out conv graphs: remove a frozen list of conv-bearing **train** graphs (mxu_frac ≥ 0.020 rule) from xla:random train, retrain baseline + channel, evaluate Top-1 on the held-out graphs | 3-graph generalization | 2 arms × 5 = 10 runs | ≈ 30 h | only if R1 finishes by ~Oct 19 | **exploratory**, held-out list frozen before training; changes the training set, so it's not comparable to the 8c numbers |
| R3 | Re-run scramble + oracle on GPU | O17 matched hardware | 2 arms × 5 = 10 runs | ≈ 30 h | yes, but competes with R1 | replication, labelled as such; the registered CPU results stay primary |
| R2 | More seeds, channel vs baseline | — | — | — | — | **not recommended.** Prohibited (`phase8c_preregistration.md:90`). Unneeded: the observed Δ needs 3 seeds [tool `required_seeds(0.3043, 0.0966)`]. Doesn't touch the 3-graph limit. |

### Recommendation

**Minimum before Oct 23: R0 (free) + R1 (pre-registered, about 60 GPU-h).** Then stop.

- **R1 is the only run that can change the headline.** It is also the only one whose absence a
  reviewer can turn into the paper's own argument: §5 says an untuned lr-1e-3 flat baseline produced a
  spurious 5/5-seed, p < .001 win. 8c's baseline is that model, at that LR, under-converged, with
  inverted conv rankings. If R1 is clean, the conv result becomes the safest finding in the paper and
  can lead. If not, the paper learns it before reviewers do.
- **R0** costs almost nothing and fixes the metric mismatch in option A's contrast.
- **R4 and R3** are worth it only if the GPU stays free after R1. Otherwise disclose (3 graphs, O17)
  and rephrase O17's "no evidence of bias".
- **If R1 can't run** (GPU contention, or no time to pre-register by ~Oct 14): **don't lead with
  convolution.** Keep option A with the xla:random like-for-like contrast and state in §7 that the 8c
  baseline was not LR-tuned.
- **Timing:** R1 can't land before the Oct 16 co-author draft. Draft §7 with a slot for it.

**Fix regardless of runs:** the number mismatches 1–6 and the stale-framing rows above. They are
text-only.

---

## Corrections after external review (2026-10-11, same day)

1. **2.8× vs 4.35×.** This audit first recommended reporting the de-duplicated 4.35×. That was wrong:
   it is a post-hoc recomputation and happens to be the larger number, so choosing it after seeing
   the data is the move the paper's audit criticises. **Headline = the original analysis with the
   counting fixed: n = 95 / 49, 2.77×, MWU p = 1.5e-7.** Unique-graph 4.35× and the per-collection
   ratios are reported as post-hoc robustness checks. The text above is corrected.
2. **Overall-τ jump goes into R1 as a report-either-way item.** Channel raises overall xla:random
   val τ by about +0.10 (0.3326 vs 0.2311). That looks like a training-dynamics change, not a
   conv-specific one. 8e (`phase8e_lr_robustness_preregistration.md`, draft) reports it at every LR
   whichever way it falls.
3. **O17 wording.** Replace "No evidence of bias" with "not tested at matched hardware" unless R3 runs.
4. **R0 is not blind.** While drafting the R0 declaration, Phase 5's cache
   (`checkpoints/phase5_mispred.json`) turned out to already hold flat vs hardware CONV Top-1 on
   xla:random (5-seed means): inception_v3 1.4433 vs 1.4389, resnet50 1.511 vs 1.5126, resnet_v1_50
   1.4227 vs 1.4803. Disclosed in `experiments/r0_conv_top1_structure_declaration.md`.
