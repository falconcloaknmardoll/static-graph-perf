# Top-K slowdown — nlp_random (exploratory, declared before computing; O9)

slowdown@K = best true runtime in the model's top-K / true best runtime (1.0 = optimal), mean over validation graphs; eval_batch 8; 5 seeds [42, 100, 123, 456, 789]. Graphs per K: K=1: 20, K=5: 20, K=10: 20, K=100: 20.

| arm | slowdown@1 | slowdown@5 | slowdown@10 | slowdown@100 |
|---|---|---|---|---|
| wide-baseline | 1.0340 ± 0.0083 | 1.0088 ± 0.0044 | 1.0064 ± 0.0028 | 1.0032 ± 0.0021 |
| hardware | 1.0530 ± 0.0123 | 1.0213 ± 0.0034 | 1.0132 ± 0.0033 | 1.0031 ± 0.0012 |

Paired Δ vs wide-baseline (negative = structured arm lands closer to optimal; paired t, two-sided, n=5):

| contrast | @1 | @5 | @10 | @100 |
|---|---|---|---|---|
| hardware − flat | +0.0190 ± 0.0188 (p=0.0865) | +0.0125 ± 0.0058 (p=0.00832) | +0.0067 ± 0.0043 (p=0.0254) | -0.0001 ± 0.0025 (p=0.948) |
