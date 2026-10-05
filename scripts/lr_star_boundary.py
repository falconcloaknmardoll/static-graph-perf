"""Apply the frozen LR* + boundary rule (6e/6f §4) to finished runs and print the boundary LR, or nothing.

LR* = the LR with the highest mean best val τ over the 5 seeds; an exact tie goes to 1e-3. A collapsed run
logs τ = +0.0000 (evaluate()'s fallback), which is the registered collapse entry (τ = 0). Prints 3e-5 if LR*
is 1e-4, 1e-2 if LR* is 3e-3, else nothing. Exits 1 if any run's log is missing or unfinished.

  python scripts/lr_star_boundary.py --log-pattern 'logs/capped/xla_default_hardware-lr{lr}_seed{seed}.log' \
      --base-log-pattern 'logs/capped/xla_default_hardware_seed{seed}.log'
"""
import argparse
import re
import sys
from statistics import mean

SEEDS = [42, 100, 123, 456, 789]
GRID = ["1e-4", "3e-4", "1e-3", "3e-3"]
BOUNDARY = {"1e-4": "3e-5", "3e-3": "1e-2"}


def best_tau(path):
    try:
        text = open(path).read()
    except FileNotFoundError:
        sys.exit(f"missing log: {path}")
    m = re.findall(r"Best val_τ = ([+-]?\d+\.\d+)", text)
    if not m:
        sys.exit(f"unfinished run: {path}")
    return float(m[-1])


ap = argparse.ArgumentParser()
ap.add_argument("--log-pattern", required=True, help="log path for the new LRs, with {lr} and {seed}")
ap.add_argument("--base-log-pattern", required=True, help="log path for the existing lr-1e-3 runs, with {seed}")
args = ap.parse_args()

means = {}
for lr in GRID:
    pat = args.base_log_pattern if lr == "1e-3" else args.log_pattern
    means[lr] = mean(best_tau(pat.format(lr=lr, seed=s)) for s in SEEDS)
top = max(means.values())
winners = [lr for lr, v in means.items() if v == top]
lr_star = "1e-3" if "1e-3" in winners else winners[0]
print(f"LR*={lr_star}", file=sys.stderr)
if lr_star in BOUNDARY:
    print(BOUNDARY[lr_star])
