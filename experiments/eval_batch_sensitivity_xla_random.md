# Eval-batch sensitivity — xla_random (post-hoc, exploratory)

Checkpoints: `checkpoints/{ablation,phase3}/<arm>-seed<s>/best_model.pt`, 5 seeds [42, 100, 123, 456, 789]. Scored with `eval_fast.evaluate_fast` (verified against `train.evaluate`). Originally scored at eval_batch 32 (Phase 2c/3); Phase 6 protocol = eval_batch 8. OOM = that eval batch does not fit in GPU memory for that arm.

| arm | τ @ b=8 | τ @ b=2 | τ @ b=32 | τ @ b=64 |
|---|---|---|---|---|
| wide-baseline | +0.2880 ± 0.0365 | +0.2741 ± 0.0472 | +0.2914 ± 0.0343 | +0.2926 ± 0.0337 |
| hierarchical | +0.2645 ± 0.0230 | +0.1870 ± 0.0187 | +0.2797 ± 0.0259 | +0.2826 ± 0.0260 |
| hardware | +0.2640 ± 0.0360 | +0.1800 ± 0.0261 | +0.2797 ± 0.0396 | OOM |

Paired Δτ vs wide-baseline (paired t, two-sided, n=5):

| contrast | b=8 | b=2 | b=32 | b=64 |
|---|---|---|---|---|
| hierarchical − flat | -0.0235 ± 0.0214 (p=0.0702) | -0.0871 ± 0.0299 (p=0.00286) | -0.0117 ± 0.0181 (p=0.221) | -0.0100 ± 0.0179 (p=0.28) |
| hardware − flat | -0.0241 ± 0.0665 (p=0.464) | -0.0941 ± 0.0691 (p=0.0383) | -0.0116 ± 0.0672 (p=0.718) | OOM |
