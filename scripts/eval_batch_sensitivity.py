"""Post-hoc sensitivity check: does paired Δτ depend on the evaluation batch size?

Every model uses BatchNorm with track_running_stats=False, so even in eval mode a
config's score is normalised with statistics from the other configs in its chunk,
and τ therefore depends on --eval-batch. The sweeps used eval_batch=8 for both arms,
so the registered comparisons are internally matched. This script re-scores the
saved best checkpoints at other chunk sizes to see whether the paired contrasts
survive. Inference only. Exploratory, not part of any pre-registration.

  python scripts/eval_batch_sensitivity.py --coll xla_default --batches 8 2 32 64
"""
import argparse, json, os, sys
import numpy as np, scipy.stats as st, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import TPUGraphDataset
from eval_fast import evaluate_fast
from model import BaselineGraphSAGE_Large
from model_phase2 import HierarchicalGraphSAGE
from model_phase3 import HardwareAwareDiffPool

ap = argparse.ArgumentParser()
ap.add_argument("--coll", default="xla_default")
ap.add_argument("--batches", type=int, nargs="+", default=[8, 2, 32, 64])
ap.add_argument("--out", default="experiments/eval_batch_sensitivity")
a = ap.parse_args()
dev = torch.device("cuda")
SEEDS = [42, 100, 123, 456, 789]
MAKERS = {
    "wide-baseline": lambda h: BaselineGraphSAGE_Large(in_channels=158, dropout=0.0),
    "hierarchical":  lambda h: HierarchicalGraphSAGE(in_channels=158, hidden_channels=h, dropout=0.0),
    "hardware":      lambda h: HardwareAwareDiffPool(in_channels=158, hidden_channels=h, dropout=0.0),
}
# xla:random predates the Phase 6 sweep: its arms live in the Phase 2c/3 directories.
if a.coll == "xla_random":
    DIRS = {"wide-baseline": "checkpoints/ablation/wide-baseline",
            "hierarchical":  "checkpoints/ablation/diffpool",
            "hardware":      "checkpoints/phase3/hardware"}
else:
    DIRS = {m: f"checkpoints/expansion/{a.coll}/{m}" for m in MAKERS}
base = "checkpoints/{ablation,phase3}" if a.coll == "xla_random" else f"checkpoints/expansion/{a.coll}"
ckpt = lambda arm, s: f"{DIRS[arm]}-seed{s}/best_model.pt"
arms = [m for m in MAKERS if all(os.path.exists(ckpt(m, s)) for s in SEEDS)]
data_root = None
res = {}          # res[arm][batch] = [tau per seed]
logged = {}
for arm in arms:
    res[arm] = {b: [] for b in a.batches}; logged[arm] = []
    for s in SEEDS:
        ck = torch.load(ckpt(arm, s), map_location="cpu", weights_only=False)
        args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
        data_root = args["data_root"]
        m = MAKERS[arm](args.get("hidden", 128)).to(dev)
        m.load_state_dict(ck["model_state"])
        logged[arm].append(float(ck["val_tau"]))
        val = TPUGraphDataset(os.path.join(data_root, "valid"))
        for b in a.batches:
            res[arm][b].append(evaluate_fast(m, val, dev, b)["val_tau"])
        print(f"{arm:14s} seed {s:3d}  logged {ck['val_tau']:+.4f}  " +
              "  ".join(f"b{b}={res[arm][b][-1]:+.4f}" for b in a.batches), flush=True)

lines = [f"# Eval-batch sensitivity — {a.coll} (post-hoc, exploratory)", "",
         f"Checkpoints: `{base}/<arm>-seed<s>/best_model.pt`, 5 seeds {SEEDS}. "
         "Scored with `eval_fast.evaluate_fast` (verified against `train.evaluate`). "
         ("Originally scored at eval_batch 32 (Phase 2c/3); Phase 6 protocol = eval_batch 8."
          if a.coll == "xla_random" else "Registered protocol = eval_batch 8."), "",
         "| arm | " + " | ".join(f"τ @ b={b}" for b in a.batches) + " |",
         "|---|" + "---|" * len(a.batches)]
for arm in arms:
    lines.append(f"| {arm} | " + " | ".join(f"{np.mean(res[arm][b]):+.4f} ± {np.std(res[arm][b], ddof=1):.4f}"
                                           for b in a.batches) + " |")
lines += ["", "Paired Δτ vs wide-baseline (paired t, two-sided, n=5):", "",
          "| contrast | " + " | ".join(f"b={b}" for b in a.batches) + " |",
          "|---|" + "---|" * len(a.batches)]
summary = {}
for arm in arms:
    if arm == "wide-baseline":
        continue
    cells = []
    for b in a.batches:
        d = np.array(res[arm][b]) - np.array(res["wide-baseline"][b])
        p = st.ttest_rel(res[arm][b], res["wide-baseline"][b]).pvalue
        cells.append(f"{d.mean():+.4f} ± {d.std(ddof=1):.4f} (p={p:.3g})")
        summary[f"{arm}-vs-flat@b{b}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "p": float(p),
                                           "per_seed": d.tolist()}
    lines.append(f"| {arm} − flat | " + " | ".join(cells) + " |")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(f"{a.out}_{a.coll}.md", "w").write("\n".join(lines) + "\n")
json.dump({"taus": {k: {str(b): v for b, v in d.items()} for k, d in res.items()},
           "logged": logged, "contrasts": summary}, open(f"{a.out}_{a.coll}.json", "w"), indent=1)
print("\n".join(lines))
