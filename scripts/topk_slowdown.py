"""Top-K slowdown for the Result I cells (objection O9). Exploratory, declared in DEFENSIBILITY.md.

slowdown@K = (best true runtime among the model's K top-scored configs) / (best true runtime),
per validation graph, then averaged over graphs; graphs with fewer than K configs are dropped
for that K. Same definition as phase4_evaluate_topk.py. Scored at eval_batch 8 (the Phase 6
protocol) with the verified eval_fast loader. Inference only.

  python scripts/topk_slowdown.py --coll xla_default
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
ap.add_argument("--coll", required=True, choices=["xla_default", "xla_random", "nlp_random"])
ap.add_argument("--eval-batch", type=int, default=8)
ap.add_argument("--out", default="experiments/topk_slowdown")
a = ap.parse_args()
dev = torch.device("cuda")
SEEDS = [42, 100, 123, 456, 789]
KS = [1, 5, 10, 100]
MAKERS = {
    "wide-baseline": lambda h: BaselineGraphSAGE_Large(in_channels=158, dropout=0.0),
    "hierarchical":  lambda h: HierarchicalGraphSAGE(in_channels=158, hidden_channels=h, dropout=0.0),
    "hardware":      lambda h: HardwareAwareDiffPool(in_channels=158, hidden_channels=h, dropout=0.0),
}
if a.coll == "xla_random":
    DIRS = {"wide-baseline": "checkpoints/ablation/wide-baseline",
            "hierarchical":  "checkpoints/ablation/diffpool",
            "hardware":      "checkpoints/phase3/hardware"}
else:
    DIRS = {m: f"checkpoints/expansion/{a.coll}/{m}" for m in MAKERS}
ckpt = lambda arm, s: f"{DIRS[arm]}-seed{s}/best_model.pt"
arms = [m for m in MAKERS if all(os.path.exists(ckpt(m, s)) for s in SEEDS)]


def slowdowns(pred, runtime):
    order = np.argsort(-pred, kind="stable")
    best = runtime.min()
    return {k: float(runtime[order[:k]].min() / best) for k in KS if len(runtime) >= k}


val = None
res = {arm: {k: [] for k in KS} for arm in arms}     # per-seed mean slowdown
n_graphs = {k: None for k in KS}
for arm in arms:
    for s in SEEDS:
        ck = torch.load(ckpt(arm, s), map_location="cpu", weights_only=False)
        args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
        if val is None:
            val = TPUGraphDataset(os.path.join(args["data_root"], "valid"))
        m = MAKERS[arm](args.get("hidden", 128)).to(dev)
        m.load_state_dict(ck["model_state"])
        _, preds = evaluate_fast(m, val, dev, a.eval_batch, return_preds=True)
        per_k = {k: [] for k in KS}
        for file_idx, p in preds.items():
            cfg = [c for f, c in val._index if f == file_idx]
            rt = np.array(val._get_handle(file_idx)["config_runtime"], dtype=np.float64)[cfg]
            for k, v in slowdowns(p, rt).items():
                per_k[k].append(v)
        for k in KS:
            res[arm][k].append(float(np.mean(per_k[k])) if per_k[k] else float("nan"))
            n_graphs[k] = len(per_k[k])
        print(f"{arm:14s} seed {s:3d}  " +
              "  ".join(f"@{k}={res[arm][k][-1]:.4f}" for k in KS), flush=True)

lines = [f"# Top-K slowdown — {a.coll} (exploratory, declared before computing; O9)", "",
         f"slowdown@K = best true runtime in the model's top-K / true best runtime (1.0 = optimal), "
         f"mean over validation graphs; eval_batch {a.eval_batch}; 5 seeds {SEEDS}. "
         f"Graphs per K: " + ", ".join(f"K={k}: {n_graphs[k]}" for k in KS) + ".", "",
         "| arm | " + " | ".join(f"slowdown@{k}" for k in KS) + " |",
         "|---|" + "---|" * len(KS)]
for arm in arms:
    lines.append(f"| {arm} | " + " | ".join(
        f"{np.nanmean(res[arm][k]):.4f} ± {np.nanstd(res[arm][k], ddof=1):.4f}" for k in KS) + " |")
lines += ["", "Paired Δ vs wide-baseline (negative = structured arm lands closer to optimal; "
          "paired t, two-sided, n=5):", "",
          "| contrast | " + " | ".join(f"@{k}" for k in KS) + " |", "|---|" + "---|" * len(KS)]
contrasts = {}
for arm in arms:
    if arm == "wide-baseline":
        continue
    cells = []
    for k in KS:
        d = np.array(res[arm][k]) - np.array(res["wide-baseline"][k])
        p = st.ttest_rel(res[arm][k], res["wide-baseline"][k]).pvalue
        cells.append(f"{d.mean():+.4f} ± {d.std(ddof=1):.4f} (p={p:.3g})")
        contrasts[f"{arm}-vs-flat@{k}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                           "p": float(p), "per_seed": d.tolist()}
    lines.append(f"| {arm} − flat | " + " | ".join(cells) + " |")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(f"{a.out}_{a.coll}.md", "w").write("\n".join(lines) + "\n")
json.dump({"slowdown": {arm: {str(k): v for k, v in d.items()} for arm, d in res.items()},
           "n_graphs": {str(k): v for k, v in n_graphs.items()}, "contrasts": contrasts,
           "eval_batch": a.eval_batch}, open(f"{a.out}_{a.coll}.json", "w"), indent=1)
print("\n".join(lines))
