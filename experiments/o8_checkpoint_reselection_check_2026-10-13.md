# O8: can the xla:random checkpoints be re-selected at eval_batch 8? (read-only check, 2026-10-13)

Scope: the 15 xla:random runs behind Result I's Phase 2c and Phase 3 contrasts. Nothing was
trained, evaluated or edited. The only computation is arithmetic on existing logs and JSON
(Task 3), and it is **post-hoc**.

**Answer: no, not without retraining.** Each run kept only `best_model.pt`. The other nine
validation-epoch checkpoints were overwritten as training went. **Recommendation: (b)**, keep
the one-sentence disclosure, and add one fact to it (§4).

---

## 1. What was saved

Run directories, from `scripts/eval_batch_sensitivity.py:35-41`, which builds
`checkpoints/{ablation,phase3}/<dir>-seed<s>/best_model.pt` (the same paths quoted at
`experiments/eval_batch_sensitivity_xla_random.md:3`):

| arm (report name) | directory prefix (`eval_batch_sensitivity.py:35-37`) | seeds |
|---|---|---|
| wide-baseline (flat) | `checkpoints/ablation/wide-baseline-seed*` | 42, 100, 123, 456, 789 |
| hierarchical | `checkpoints/ablation/diffpool-seed*` | same |
| hardware | `checkpoints/phase3/hardware-seed*` | same |

### 1a. Checkpoint files on disk

Each of the 15 directories holds exactly two files: `best_model.pt` and `run.log` (`ls -la`,
2026-10-13). There is no per-epoch checkpoint and no last-epoch checkpoint.

| arm | `best_model.pt` size (bytes, all 5 seeds) | `run.log` |
|---|---|---|
| wide-baseline | 2,858,741 | 8,502–8,569 B |
| diffpool (hierarchical) | 2,872,357 | 8,461–8,625 B |
| hardware | 2,878,709 | 8,527–8,632 B |

For example, `checkpoints/ablation/wide-baseline-seed42/`:
```
-rw-rw-r-- 1 user user 2858741 2026-06-22 03:24 best_model.pt
-rw-rw-r-- 1 user user    8502 2026-06-22 03:24 run.log
```
The only other files in the parent directories are `checkpoints/ablation/results_summary.txt`
and `checkpoints/phase3/results_summary.txt`. A search of `wandb/` turned up no `.pt` files.
The wandb sync line confirms nothing else was uploaded: "Synced 5 W&B file(s), 0 media file(s),
0 artifact file(s)" (`checkpoints/ablation/wide-baseline-seed42/run.log:102`).

`best_model.pt` is overwritten every time validation improves (`train.py:684-692`:
`if val_stats["val_tau"] > best_tau: ... torch.save(`). The logs show the same behaviour with
`← saved` marks, e.g. `wide-baseline-seed42/run.log:27,32,37,42,47,62,72`.

### 1b. Per-epoch validation τ log

Every `run.log` has one line per validation epoch. Validation ran every 5 epochs (`eval_every: 5`),
so each run has 10 values, at epochs 5, 10, …, 50 (e.g. `wide-baseline-seed42/run.log:27-72`).
All 10 were computed at **eval_batch 32**, according to the saved args (1c). The same history went
to wandb (`run.log:86`, `val_tau ▁▂▃▃▃▃▃▇▇█`, 10 points).

The best epoch per run (`run.log` "Done." line, e.g. `wide-baseline-seed42/run.log:106`):

| arm | 42 | 100 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| wide-baseline | 50 | 50 | 50 | **45** | 50 |
| hierarchical | 50 | 50 | 50 | **45** | 50 |
| hardware | 50 | 50 | 50 | **45** | 50 |

**12/15 selected checkpoints are the final epoch.** The 3 exceptions are all seed 456. For those
three runs, the epoch-50 weights no longer exist.

### 1c. Training args stored in the checkpoints

Every `best_model.pt` is a dict with keys `['epoch', 'model_state', 'opt_state', 'val_tau', 'args']`.
I loaded all 15 on the CPU:

```
checkpoints/ablation/wide-baseline-seed42/best_model.pt  epoch=50 val_tau=0.3193 eval_batch=32 eval_every=5 lr=0.001 model=wide-baseline
checkpoints/ablation/diffpool-seed456/best_model.pt      epoch=45 val_tau=0.2704 eval_batch=32 eval_every=5 lr=0.001 model=hierarchical
checkpoints/phase3/hardware-seed456/best_model.pt        epoch=45 val_tau=0.2814 eval_batch=32 eval_every=5 lr=0.001 model=hardware
(all 15: eval_batch=32, eval_every=5, lr=0.001, epochs=50; epoch=45 only for the three seed-456 runs)
```
The full `args` of `wide-baseline-seed42` are: `configs_per_batch 8, dropout 0.0, entropy_weight 0.01,
epochs 50, eval_batch 32, eval_every 5, grad_clip 0.1, hidden 128, lp_weight 0.1, lr 0.001,
max_configs_per_graph 128, weight_decay 0.0001, data_root .../layout/xla/random`.

The checkpoints store **no scheduler state and no RNG state**. `train.py:606-614` starts a
"Fresh LR schedule" when a run resumes.

---

## 2. Is re-selection at batch 8 possible?

**Not as an eval-only step.** Re-selecting means scoring all 10 validation-epoch checkpoints per
run at batch 8. Only the selected one exists (§1a).

**Retraining cost**, from the original logs. These timings are from GPU runs ("Device : cuda",
`wide-baseline-seed42/run.log:9`):
- Phase 2c (10 runs): `checkpoints/ablation/results_summary.txt:1` starts at Jun 22 01:08. The
  last `best_model.pt`/`results_summary.txt` mtime is Jun 22 23:14, so ≈ **22 h**.
- Phase 3 (5 runs): `checkpoints/phase3/results_summary.txt:1` starts at Jun 23 00:00. The last
  mtime is Jun 23 14:07, so ≈ **14 h**.
- Total ≈ **36 GPU-hours** (≈ 2.4 h per run, including the in-training validation). Re-running
  with a per-epoch save (or validation at batch 8) adds little to that.

Retraining would not answer the question as asked. No RNG state was saved (§1c), and the June
software stack has not been shown to reproduce bit-for-bit (I did not test this). Retraining
would therefore almost certainly produce **new models**, not a re-selection among the old ones. It would replace
every xla:random number in Result I, which makes it a new experiment that needs its own
declaration.

**Partial options considered:**
- *"Best vs last" comparison.* There is nothing to compare: for 12/15 runs best = last, and for the
  other 3 the last-epoch weights are gone.
- *Resume the three seed-456 runs from epoch 45 for 5 more epochs to recreate epoch 50.* The opt
  state exists, but there is no scheduler or RNG state (`train.py:606-614`), so the result would be
  a different epoch-50 model. That is still retraining, and it doesn't answer the question.
- *Restrict re-selection to the epochs that exist.* Each run has exactly one candidate, so the
  procedure is trivial and the selection stays the same.

---

## 3. Would it matter? (post-hoc, existing data only)

**3a. Batch-8 vs batch-32 τ for the same checkpoints**
(`experiments/eval_batch_sensitivity_xla_random.json`, `taus.<arm>.8` minus `taus.<arm>.32`):

| arm | shift per seed (42, 100, 123, 456, 789) | mean | SD | range |
|---|---|---|---|---|
| wide-baseline | −0.0030, −0.0083, −0.0018, −0.0023, −0.0013 | −0.0033 | 0.0028 | 0.0070 |
| hierarchical | −0.0180, −0.0117, −0.0141, −0.0128, −0.0189 | −0.0151 | 0.0032 | 0.0072 |
| hardware | −0.0189, −0.0206, −0.0120, −0.0149, −0.0123 | −0.0158 | 0.0039 | 0.0086 |

The batch-8 penalty depends mainly on the arm, not on the run: hierarchical and hardware lose
about 0.015, flat about 0.003. That arm-level difference produces most of the b32 → b8 change in
the contrasts (−0.0117 → −0.0235; −0.0116 → −0.0241, `eval_batch_sensitivity_xla_random.md:15-16`).
Re-selection cannot remove it unless earlier checkpoints of the same arm were much less sensitive
to batch size, and nothing in the data suggests that.

**3b. How close each selection was at batch 32** (best vs runner-up validation epoch, from `run.log`):

| arm | 42 | 100 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| wide-baseline | .0428 (ep40) | **.0067** (ep45) | .0127 (ep45) | .0096 (ep50) | .0437 (ep45) |
| hierarchical | .0312 (ep45) | **.0031** (ep45) | .0215 (ep40) | .0143 (ep50) | .0121 (ep45) |
| hardware | .0450 (ep45) | **.0045** (ep45) | .0496 (ep45) | .0147 (ep50) | **.0039** (ep45) |

The heuristic is post-hoc and cannot be checked, because the runner-up weights are gone. It
assumes two checkpoints from the same run differ in batch sensitivity by no more than different
seeds do (range ≤ 0.009). Under that assumption, only the four bold runs could flip: flat/hier/hw
seed 100 and hw seed 789. The seed-456 selections (margins 0.010–0.015) would hold.

**3c. Bound on the conclusion (post-hoc).**
- Selecting the maximum at batch 8 can only **raise** a run's batch-8 τ. A flip can raise it by at
  most about (shift range − margin): ≤ 0.0003 for flat s100, ≤ 0.004 for hier s100, ≤ 0.004 for
  hw s100, ≤ 0.005 for hw s789.
- hier − flat at b8: mean −0.0235, SD 0.0214, t = −2.46 (df 4, critical value 2.78). Reaching
  p < .05 at the same SD would take a mean of about −0.0266, a −0.015 move at one seed. The only
  move that pushes in that direction is flat s100 rising by ≤ 0.0003. Moves on the hierarchical
  side push the mean **toward zero**. Under this heuristic, p = .07 cannot become significant.
- hw − flat at b8: p = .46 (per-seed SD 0.0665). Changes of a few thousandths cannot make it
  significant, and they would move the mean toward zero anyway.
- **Conclusion:** both contrasts are non-significant at b32 and at b8. Under the stated heuristic,
  re-selecting checkpoints at b8 could not change either verdict or the sign of either mean. The
  heuristic rests on an assumption the repo cannot test (§2), so treat this as support for the
  disclosure, not as evidence.

---

## 4. Recommendation: **(b) keep the one-sentence disclosure**

Why:
1. Re-selection needs retraining (§1a, §2). About 36 GPU-hours would produce new models rather than
   a re-selection, so it would be a new experiment and would need a new declaration.
2. It does not fit the queue without risk. As of 2026-10-13 23:35:
   - Queue l is running R4 (`logs/gpu_queue_2026-10-11l.log`). Run 3 of 10 started at 21:48, and
     the completed runs took about 1 h 55 m each, so R4 ends around **Oct 14 midday**.
   - Queue m (R3, 5 runs, `scripts/run_gpu_queue_2026-10-11m.sh`) waits on k/l.
   - Queue n (R0, eval-only, `scripts/run_gpu_queue_2026-10-13n.sh`) waits on k/l/m. R0's
     duration on the GPU is unknown; its CPU attempt was stopped after 41 h (commit 44dbd22).
   - 8e: the grid and boundary steps are done (`logs/gpu_queue_2026-10-11l.log`, last
     `DONE ... mxu-channel-lr3e-5-seed789` at 10-13 18:03).
   - Placed after R0, 36 GPU-hours would end around Oct 17–18 at the earliest. That leaves no slack
     before the Oct 23 cutoff if anything in R3 or R0 slips. Put before them, it would delay R3 and R0.
3. Under the post-hoc heuristic, re-selection would not change either verdict (§3c).

**Facts the disclosure can draw on** (for the first author's sentence; I have not drafted prose):
- The checkpoints were selected on validation τ at eval_batch 32, every 5 epochs (saved `args`, §1c).
- 12/15 selected checkpoints are the final epoch (50). The other 3 are epoch 45, all seed 456 (§1b).
  So for most runs, the selection rule and its batch size did not affect which model was chosen.
- The batch-8 penalty depends on the arm (hier/hw ≈ −0.015, flat ≈ −0.003, §3a), not on which
  checkpoint was picked.

No declaration is drafted, since (a) is not recommended.

### Repo note
DEFENSIBILITY O8 (`DEFENSIBILITY.md:72`) says "Disclose: best-checkpoint selection for xla:random
was done at b=32." This check supports that disclosure. If the first author wants it logged, the change is
one line in the O8 evidence column pointing to this file. I have not made that edit.
