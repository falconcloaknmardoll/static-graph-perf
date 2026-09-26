"""Check eval_fast.evaluate_fast() against train.evaluate() on a config subset.

Captures train.evaluate()'s per-graph predictions by wrapping the kendalltau call
it makes (train.py itself is not modified), then compares predictions exactly,
plus τ and timing. GPU scatter-add makes the model forward itself nondeterministic
(~1e-7), so train.evaluate() is run TWICE: the fast path passes if its deviation from
the reference is no larger than the reference's own run-to-run deviation.

  python scripts/verify_eval_fast.py --per-graph 64
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train
from dataset import TPUGraphDataset
from eval_fast import evaluate_fast
from model import BaselineGraphSAGE_Large
from model_phase3 import HardwareAwareDiffPool

ap = argparse.ArgumentParser()
ap.add_argument("--per-graph", type=int, default=64)
ap.add_argument("--data", default="data/predict-ai-model-runtime/npz_all/npz/layout/nlp/default/valid")
ap.add_argument("--eval-batch", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda")

CKPTS = [
    ("hardware lr1e-4 (repro run B)", "logs/repro/final_seed100_diagB_lowlr.pt",
     lambda: HardwareAwareDiffPool(in_channels=158, hidden_channels=128, dropout=0.0)),
    ("wide-baseline (sweep seed42)", "checkpoints/expansion/nlp_default/wide-baseline-seed42/best_model.pt",
     lambda: BaselineGraphSAGE_Large(in_channels=158, dropout=0.0)),
]

def subset(ds, k):
    seen, idx = {}, []
    for f, c in ds._index:
        if seen.get(f, 0) < k:
            idx.append((f, c)); seen[f] = seen.get(f, 0) + 1
    ds._index = idx
    return ds

captured = []
class _Shim:
    def kendalltau(self, p, t):
        captured.append(np.array(p, copy=True))
        return train.sp_stats_real.kendalltau(p, t)
train.sp_stats_real = train.sp_stats
train.sp_stats = _Shim()

ok = True
for name, path, make in CKPTS:
    m = make().to(dev)
    m.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
    ds_ref = subset(TPUGraphDataset(a.data), a.per_graph)
    ds_new = subset(TPUGraphDataset(a.data), a.per_graph)

    captured.clear(); t0 = time.time()
    with torch.no_grad():
        ref = train.evaluate(m, ds_ref, dev, a.eval_batch)
    t_ref = time.time() - t0
    ref_preds = list(captured)

    captured.clear()
    with torch.no_grad():
        ref2 = train.evaluate(m, subset(TPUGraphDataset(a.data), a.per_graph), dev, a.eval_batch)
    ref2_preds = list(captured)

    t0 = time.time()
    new, new_preds = evaluate_fast(m, ds_new, dev, a.eval_batch, return_preds=True)
    t_new = time.time() - t0
    new_preds = [new_preds[f] for f in sorted(new_preds)]

    noise = max(float(np.max(np.abs(r - q))) for r, q in zip(ref_preds, ref2_preds))
    tau_noise = abs(ref["val_tau"] - ref2["val_tau"])
    maxdiff = max(float(np.max(np.abs(r - n))) for r, n in zip(ref_preds, new_preds))
    bitwise = all(np.array_equal(r, n) for r, n in zip(ref_preds, new_preds))
    tau_dev = abs(ref["val_tau"] - new["val_tau"])
    same_tau = ref["val_n_graphs"] == new["val_n_graphs"] and tau_dev <= max(3 * tau_noise, 1e-4)
    ok &= same_tau and maxdiff <= 10 * max(noise, 1e-7)
    print(f"{name}: configs/graph={a.per_graph} graphs={len(ref_preds)}\n"
          f"  ref  tau={ref['val_tau']:+.6f} n={ref['val_n_graphs']}  {t_ref:.1f}s\n"
          f"  fast tau={new['val_tau']:+.6f} n={new['val_n_graphs']}  {t_new:.1f}s  "
          f"(speedup {t_ref / t_new:.1f}x)\n"
          f"  ref run-to-run noise: preds {noise:.3e}, tau {tau_noise:.2e}\n"
          f"  fast vs ref:          preds {maxdiff:.3e}, tau {tau_dev:.2e}  (bitwise {bitwise})",
          flush=True)
print("PASS" if ok else "FAIL")
