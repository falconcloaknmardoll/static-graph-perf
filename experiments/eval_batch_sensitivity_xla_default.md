# Eval-batch sensitivity — xla_default (post-hoc, exploratory)

Checkpoints: `checkpoints/expansion/xla_default/<arm>-seed<s>/best_model.pt`, 5 seeds [42, 100, 123, 456, 789]. Scored with `eval_fast.evaluate_fast` (verified against `train.evaluate`). Registered protocol = eval_batch 8. OOM = that eval batch does not fit in GPU memory for that arm.

| arm | τ @ b=8 | τ @ b=2 | τ @ b=32 | τ @ b=64 |
|---|---|---|---|---|
| wide-baseline | +0.0327 ± 0.0135 | +0.0304 ± 0.0104 | +0.0330 ± 0.0138 | +0.0330 ± 0.0137 |
| hierarchical | +0.1484 ± 0.0276 | +0.1070 ± 0.0169 | +0.1537 ± 0.0322 | +0.1590 ± 0.0327 |
| hardware | +0.1977 ± 0.0133 | +0.1531 ± 0.0155 | +0.1913 ± 0.0214 | OOM |

Paired Δτ vs wide-baseline (paired t, two-sided, n=5):

| contrast | b=8 | b=2 | b=32 | b=64 |
|---|---|---|---|---|
| hierarchical − flat | +0.1157 ± 0.0358 (p=0.00194) | +0.0765 ± 0.0246 (p=0.00224) | +0.1206 ± 0.0409 (p=0.00275) | +0.1260 ± 0.0405 (p=0.00224) |
| hardware − flat | +0.1650 ± 0.0133 (p=9.95e-06) | +0.1227 ± 0.0163 (p=7.28e-05) | +0.1583 ± 0.0191 (p=4.97e-05) | OOM |
