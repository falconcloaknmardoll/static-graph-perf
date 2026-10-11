"""Apply 8e's frozen LR* + boundary rule (phase8e_lr_robustness_preregistration.md §4) for one arm.

LR*_arm = the LR in {1e-4, 3e-4, 1e-3} (plus any boundary LR already run) with the highest mean best
val tau (overall, xla/random/valid) over the 5 seeds; an exact tie goes to 1e-3. lr 1e-3 = the 8c run.
Prints the boundary LR to run (3e-5 if LR* = 1e-4, 3e-3 if LR* = 1e-3), or nothing.
Exits 1 if any required log is missing or unfinished.
  python scripts/phase8e_lr_star.py --arm baseline
"""
import argparse
import os
import re
import sys
from statistics import mean

SEEDS = [42, 100, 123, 456, 789]
GRID = ["1e-4", "3e-4", "1e-3"]
BOUNDARY = {"1e-4": "3e-5", "1e-3": "3e-3"}


def log_path(arm, lr, seed):
    if lr == "1e-3":
        return f"logs/capped/xla_random_mxu-{arm}_seed{seed}.log"
    return f"logs/capped/xla_random_mxu-{arm}-lr{lr}_seed{seed}.log"


def best_tau(path):
    try:
        m = re.findall(r"Best val_τ = ([+-]?\d+\.\d+)", open(path).read())
    except FileNotFoundError:
        sys.exit(f"missing log: {path}")
    if not m:
        sys.exit(f"unfinished run: {path}")
    return float(m[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "channel"])
    a = ap.parse_args()
    grid = GRID + [b for b in BOUNDARY.values() if os.path.exists(log_path(a.arm, b, SEEDS[0]))]
    means = {lr: mean(best_tau(log_path(a.arm, lr, s)) for s in SEEDS) for lr in grid}
    best = max(means.values())
    tied = [lr for lr in grid if means[lr] == best]
    lr_star = "1e-3" if "1e-3" in tied else tied[0]
    print(f"{a.arm}: " + ", ".join(f"{lr}={m:.4f}" for lr, m in means.items()) + f" -> LR*={lr_star}",
          file=sys.stderr)
    b = BOUNDARY.get(lr_star)
    if b and b not in grid:
        print(b)
