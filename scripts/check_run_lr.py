"""Exit 0 iff a train.py log's epoch-1 LR matches --lr under train.py's schedule.

train.py prints "Fresh LR schedule" only when resuming, so fresh runs are checked via the
epoch-1 line instead: cosine decay lr -> lr/100 over 50 epochs, one step taken at epoch 1.
  python scripts/check_run_lr.py <log> <lr> [epochs]
"""
import math, re, sys

log, lr = sys.argv[1], float(sys.argv[2])
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
m = re.search(r"^Epoch\s+1/\d+\s.*?\blr=([0-9.e+-]+)", open(log, errors="replace").read(), re.M)
if not m:
    sys.exit(f"no epoch-1 line in {log}")
eta_min = lr * 0.01
expected = eta_min + (lr - eta_min) * (1 + math.cos(math.pi / epochs)) / 2
got = float(m.group(1))
ok = abs(got - expected) / expected < 0.01
print(f"epoch-1 lr {got:.3e} vs expected {expected:.3e} for --lr {lr:g}: {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
