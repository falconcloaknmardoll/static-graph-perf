# Top-K slowdown — xla_default (exploratory, declared before computing; O9)

slowdown@K = best true runtime in the model's top-K / true best runtime (1.0 = optimal), mean over validation graphs; eval_batch 8; 5 seeds [42, 100, 123, 456, 789]. Graphs per K: K=1: 7, K=5: 7, K=10: 7, K=100: 7.

| arm | slowdown@1 | slowdown@5 | slowdown@10 | slowdown@100 |
|---|---|---|---|---|
| wide-baseline | 1.1521 ± 0.0179 | 1.0999 ± 0.0364 | 1.0724 ± 0.0036 | 1.0082 ± 0.0005 |
| hierarchical | 1.0062 ± 0.0019 | 1.0031 ± 0.0019 | 1.0029 ± 0.0017 | 1.0001 ± 0.0000 |
| hardware | 1.0090 ± 0.0077 | 1.0017 ± 0.0015 | 1.0016 ± 0.0016 | 1.0005 ± 0.0006 |

Paired Δ vs wide-baseline (negative = structured arm lands closer to optimal; paired t, two-sided, n=5):

| contrast | @1 | @5 | @10 | @100 |
|---|---|---|---|---|
| hierarchical − flat | -0.1459 ± 0.0174 (p=4.79e-05) | -0.0968 ± 0.0363 (p=0.00398) | -0.0695 ± 0.0045 (p=4.05e-06) | -0.0081 ± 0.0005 (p=3.48e-06) |
| hardware − flat | -0.1431 ± 0.0220 (p=0.00013) | -0.0983 ± 0.0375 (p=0.00424) | -0.0708 ± 0.0025 (p=3.53e-07) | -0.0077 ± 0.0004 (p=2.63e-06) |
