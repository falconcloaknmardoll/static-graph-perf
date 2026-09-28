# Top-K slowdown — xla_random (exploratory, declared before computing; O9)

slowdown@K = best true runtime in the model's top-K / true best runtime (1.0 = optimal), mean over validation graphs; eval_batch 8; 5 seeds [42, 100, 123, 456, 789]. Graphs per K: K=1: 7, K=5: 7, K=10: 7, K=100: 7.

| arm | slowdown@1 | slowdown@5 | slowdown@10 | slowdown@100 |
|---|---|---|---|---|
| wide-baseline | 1.1951 ± 0.1097 | 1.1026 ± 0.0644 | 1.0787 ± 0.0485 | 1.0417 ± 0.0417 |
| hierarchical | 1.1851 ± 0.0546 | 1.1119 ± 0.0306 | 1.0861 ± 0.0114 | 1.0550 ± 0.0252 |
| hardware | 1.2006 ± 0.0530 | 1.0995 ± 0.0360 | 1.0697 ± 0.0355 | 1.0204 ± 0.0253 |

Paired Δ vs wide-baseline (negative = structured arm lands closer to optimal; paired t, two-sided, n=5):

| contrast | @1 | @5 | @10 | @100 |
|---|---|---|---|---|
| hierarchical − flat | -0.0100 ± 0.0946 (p=0.825) | +0.0094 ± 0.0784 (p=0.803) | +0.0073 ± 0.0423 (p=0.719) | +0.0134 ± 0.0381 (p=0.477) |
| hardware − flat | +0.0055 ± 0.0869 (p=0.894) | -0.0031 ± 0.0786 (p=0.934) | -0.0090 ± 0.0610 (p=0.757) | -0.0213 ± 0.0570 (p=0.451) |
