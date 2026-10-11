# Top-K slowdown — xla_default_tuned (exploratory, declared before computing; O9)

slowdown@K = best true runtime in the model's top-K / true best runtime (1.0 = optimal), mean over validation graphs; eval_batch 8; 5 seeds [42, 100, 123, 456, 789]. Graphs per K: K=1: 7, K=5: 7, K=10: 7, K=100: 7.

| arm | slowdown@1 | slowdown@5 | slowdown@10 | slowdown@100 |
|---|---|---|---|---|
| wide-baseline | 1.0176 ± 0.0091 | 1.0060 ± 0.0077 | 1.0013 ± 0.0011 | 1.0003 ± 0.0004 |
| hardware | 1.0042 ± 0.0040 | 1.0005 ± 0.0004 | 1.0005 ± 0.0004 | 1.0001 ± 0.0000 |

Paired Δ vs wide-baseline (negative = structured arm lands closer to optimal; paired t, two-sided, n=5):

| contrast | @1 | @5 | @10 | @100 |
|---|---|---|---|---|
| hardware − flat | -0.0134 ± 0.0128 (p=0.0791) | -0.0055 ± 0.0074 (p=0.176) | -0.0008 ± 0.0012 (p=0.203) | -0.0002 ± 0.0004 (p=0.429) |
