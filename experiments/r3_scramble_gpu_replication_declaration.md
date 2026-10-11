# R3 — P4 (channel vs scramble) at matched hardware: GPU re-run of the scramble arm (replication)

> **Status: FROZEN ON COMMIT, 2026-10-11.** the first author approved queueing it in session (2026-10-11). It runs
> third, after 8e and R4 (queue m waits for queue l), so any change is a dated amendment made
> before the first R3 run starts. This is a **replication**: the registered P4 result
> (`phase8c_control_results.md`, scramble trained on CPU) stays primary. R3 is reported beside it.
> Origin: DEFENSIBILITY O17.

## 0. Observation status

- **Known:** 8c baseline/channel (GPU) and scramble (CPU) results; P4 Δ −0.2442, p = .0063; scramble
  vs baseline Δ −0.0601, p = .30; scramble's OTHER placebo +0.0079, p = .028 (the only placebo
  pairing a CPU arm with a GPU arm).
- **Never run:** the scramble arm on GPU.

## 1. Question

Does P4 (channel significantly beats scramble on CONV Top-1) hold when scramble trains on the same
hardware as channel? And how large is the CPU-vs-GPU difference for an otherwise identical arm?

## 2. Design (fixed now)

- **Arm:** `scramble`, exactly as 8c Amendment A2 (`phase8c_preregistration.md:117–142`):
  `train_8c.py --mxu-arm scramble`, same per-graph deterministic permutation.
- **Recipe:** 8c's registered recipe, **lr 1e-3** (P4 was registered at lr 1e-3, and this replicates
  it, not 8e). 50 epochs, capped harness (num_workers 0), **GPU**, seeds {42, 100, 123, 456, 789}.
- **Paths:** checkpoints `checkpoints/r3/xla_random/mxu-scramble-gpu-seed<s>`, logs
  `logs/capped/xla_random_mxu-scramble-gpu_seed<s>.log`. The tag differs from 8c's so the CPU logs
  are never touched.
- **Metric:** CONV Top-1 slowdown on 8c's frozen split, `phase8c_evaluate.py`'s definition,
  **eval batch 32** (as P4), with batch 8 as sensitivity.
- **Contrasts (paired by seed, two-sided t, plus Wilcoxon, bootstrap CI, d and MDE):**
  1. **R3-P4:** channel (GPU, 8c) − scramble-GPU, CONV. This is the matched-hardware replication.
  2. scramble-GPU − baseline (GPU, 8c), CONV, and its OTHER placebo.
  3. **Hardware effect:** scramble-GPU − scramble-CPU, CONV and OTHER, for the same arm and seeds.
     It measures the CPU/GPU offset directly. Descriptive.

## 3. How it will be used (fixed now)

| R3-P4 outcome | Wording |
|---|---|
| Δ < 0, p < .05 | "P4 replicates at matched hardware." O17 → answered for P4. Report both numbers. |
| Δ < 0, p ≥ .05 | "P4 holds as registered but did not replicate significantly at matched hardware (MDE …)." O17 stays disclosed, with this result. |
| Δ ≥ 0 | "P4 does not survive matched hardware." §7's "real information, not capacity" claim is withdrawn and restated as untested. |

Contrast 3 is reported whichever way it falls. If the OTHER placebo moves on GPU too, the
CPU-offset explanation for the original wobble is dropped. **Cutoff:** all 5 runs must finish by
2026-10-23 23:59 ET, otherwise R3 is not reported. No seeds added.

## 4. Budget and order

5 runs × ≈ 3 h ≈ 15 GPU-h. Queue `scripts/run_gpu_queue_2026-10-11m.sh` waits for queue l
(8e boundary + R4). It is the first item to drop if GPU time runs short. The 8d oracle is
**not** re-run: 8d is descriptive only, and P5 is not confirmed on any hardware.
