# Backburner: deferred compute (decided 2026-09-29)

**Decision (the first author, 2026-09-29):** the GPU is shared with the lab, so only the *musts* run now
(`scripts/run_gpu_queue_2026-09-29g.sh`: Phase 6e, then O9 Top-K nlp:random). Everything below
is deferred, not cancelled. If a large block of GPU time opens up in October, pick from this list.
The decision was made **before any data from these runs exists**, so deferring them is not a
selective-reporting choice.

**Pause switch:** `touch PAUSE_GPU_QUEUE` in the repo root makes the queue wait after the run in
progress; `rm PAUSE_GPU_QUEUE` resumes. The musts-only queue logs to `logs/gpu_queue_2026-09-29g.log`.

**Measured footprint (2026-09-29, xla:default flat run):** ~7.4 GB of the 16 GB GPU, ~33% util,
~72 s/epoch on a quiet machine (≈1–1.5 h per 50-epoch run incl. evals). Lab-mates can co-run a
job of up to ~8 GB alongside it.

| # | Item | Priority | GPU cost | Deadline / trigger | Paper impact if never run |
|---|---|---|---|---|---|
| B0 | 6e boundary step | **Conditional MUST** | 5 runs, ~5–8 h | Only if LR\* = 1e-4 or 3e-3 | Not optional when triggered: the 6e prereg (§4 boundary rule) requires it |
| B1 | nlp:default amended sweep (6d §10.5, Amendment 2) — **running, 10/13 done Oct 10** | High among optional | 13 runs, ~2.5–3.5 days | **All 13 `.done` by 2026-10-23 23:59 ET**, so start by ~Oct 18 | Report §10.4 only (registered training failure). Result I stays scoped to xla. |
| B2 | O17 matched-hardware channel arm | Low | CPU, several days (5 runs) | Any time before submission | Keep the O17 disclosure as written |
| B3 | O11 locality test | Low | Unknown; needs a design and a pre-registration first | Only after 6e confirms the win | Keep the O11 limitation as written |

## B0: Phase 6e boundary step (conditional must)
- **✅ TRIGGERED and LAUNCHED 2026-10-05 13:31** (LR\* = 1e-4): queue (h),
  `scripts/run_gpu_queue_2026-10-05h.sh`, log `logs/gpu_queue_2026-10-05h.log`. Followed by
  queue (i), Phase 6f (`scripts/run_gpu_queue_2026-10-05i.sh`, 15 hardware runs, ~38 h).
- **Trigger:** when 6e is evaluated, LR\* (flat's best LR over {1e-3, 1e-4, 3e-4, 3e-3}) sits at
  an edge (1e-4 or 3e-3).
- **Run:** one half-decade step past that edge (3e-5 or 1e-2), seeds {42,100,123,456,789},
  same protocol. LR\* is then re-selected over the enlarged grid. No other LR may be added.
- **Source:** `experiments/phase6e_flat_lr_preregistration.md` §4 (boundary rule), §8 item 2.
- **Launch:** add a loop to a new queue script modelled on the 6e loop in
  `scripts/run_gpu_queue_2026-09-29g.sh`, with `LR` set to the boundary value.

## B1: nlp:default amended sweep
- **✅ CHAINED 2026-10-05 (the first author):** queue (j), `scripts/run_gpu_queue_2026-10-05j.sh`, log
  `logs/gpu_queue_2026-10-05j.log`. Starts automatically when queue (i) (6f) exits, after the 6f
  boundary step if that fires. Started Oct 8 10:43 (after the 6f boundary step); **10/13 `.done` by Oct 10 15:00**, the last
  flat@1e-3 runs finish ~Oct 10 night, well before the registered Oct 23 cutoff. Evaluate under 6d §10.5 next.
  Checkpoints: lr-1e-4 runs reach `checkpoints/expansion/nlp_default_lr1e-4/` through the symlink `checkpoints/_stage_lr1e-4/nlp_default`.
- **What:** 10 fresh runs at lr 1e-4 (flat and hardware × 5 seeds) → `checkpoints/expansion/nlp_default_lr1e-4/`,
  plus flat@1e-3 seeds 123/456/789 → `checkpoints/expansion/nlp_default/`. All use `--fast-eval`.
- **The three flat@1e-3 seeds are not optional within B1.** §10.5's confound check requires
  hardware@1e-4 to also beat flat@1e-3. Without them, A-P1 cannot be stated as supported.
  Run all 13 or none.
- **Why it's worth running if time allows:** it fills the missing cell of the 2×2 and tests
  whether distribution-dependence carries beyond xla. That is the paper's generality weak spot (O4, O10).
- **Why it's deferrable:** the paper is already scoped to xla, and "not finished by the cutoff →
  report §10.4 only" is a registered outcome (Amendment 2, §11.2). No partial-seed result may be
  reported, so starting late and missing the cutoff wastes the GPU time. **Don't start it
  unless all 13 runs can finish by Oct 23.**
- **Cost:** nlp:default epochs took ~220–330 s under heavy CPU contention in September;
  `--fast-eval` removes the multi-hour evals. Estimate ~4–6 h per run, ~2.5–3.5 days total.
- **Launch:** the exact loop is in `scripts/run_gpu_queue_2026-09-27f.sh` (section
  "6d §10.5: amended nlp:default sweep", including the `_stage_lr1e-4` symlink). Copy it into a
  new queue script that has the pause switch.
- **Source:** `experiments/phase6d_nlp_default_preregistration.md` §10.5, §11.

## B2: O17 matched-hardware re-run (optional)
- Retrain the 8c channel arm (5 seeds) on CPU, so that P4 (channel vs scramble) compares arms
  trained on the same hardware. CPU-only, so it doesn't compete for the GPU, but the CPU is also shared.
- **Source:** `DEFENSIBILITY.md` O17. Author's call; the disclosure alone is sufficient.

## B3: O11 locality test (optional)
- **2026-10-05: trigger not met.** Its precondition was "6e confirms the win"; 6e's P1 failed.
  Do not run.
- Tests the untested hypothesis that hierarchy wins on xla:default because it sees non-local
  layout effects that the flat model's ~3-hop config readout cannot.
- Requires a design and a pre-registration **before** any run, and only makes sense if 6e confirms the win.
- **Source:** `DEFENSIBILITY.md` O11.

## Log
- 2026-09-29 16:39: machine reboot killed queue (f) during 6e `wide-baseline-lr3e-4-seed42`
  (epoch 29/50; log last written 15:35). The partial run and its log were moved to
  `checkpoints/flatlr/_aborted/xla_default_wide-baseline-lr3e-4-seed42_reboot-2026-09-29/`,
  and the run restarts fresh (no resume). Disclosure: that log shows one interim val τ
  (epoch 25), which was seen while checking queue status. It is not used for anything.
  The rerun replaces it, and LR\* is selected only from completed runs, per the prereg.
- 2026-09-29 17:41: queue (g) launched (musts only, with the pause switch).
