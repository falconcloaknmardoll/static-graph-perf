"""R0 — CONV-subset Top-1 of the structured arms on xla:random (exploratory, eval-only).

Frozen in experiments/r0_conv_top1_structure_declaration.md (44267c7) + Amendment 1 (5defe35):
existing Phase 2c/3 xla:random checkpoints (flat, hierarchical, hardware; lr 1e-3, 5 seeds),
scored on 8c's frozen split (CONV 3 / OTHER 4 graphs, both xla valid sets) with 8c's Top-1
definition (argmax of predictions over a graph's configs; runtime / best). Primary eval batch 32
(= 8c), batch 8 as sensitivity. Contrasts: hierarchical - flat and hardware - flat, paired by seed.
  python scripts/r0_conv_top1_eval.py   -> experiments/r0_conv_top1_results.{md,json}
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import TPUGraphDataset                                   # noqa: E402
from model import BaselineGraphSAGE_Large                             # noqa: E402
from model_phase2 import HierarchicalGraphSAGE                        # noqa: E402
from model_phase3 import HardwareAwareDiffPool                        # noqa: E402
from train import _forward                                            # noqa: E402
from phase8c_evaluate import CONV_GRAPHS, OTHER_GRAPHS, COLLECTIONS, DATA, paired_stats  # noqa: E402
from power_analysis import min_detectable_effect                      # noqa: E402

SEEDS = [42, 100, 123, 456, 789]
BATCHES = [32, 8]                       # 32 = primary (8c's), 8 = sensitivity (Amendment 1)
ARMS = {"flat":         ("checkpoints/ablation/wide-baseline",
                         lambda h: BaselineGraphSAGE_Large(in_channels=158, dropout=0.0)),
        "hierarchical": ("checkpoints/ablation/diffpool",
                         lambda h: HierarchicalGraphSAGE(in_channels=158, hidden_channels=h, dropout=0.0)),
        "hardware":     ("checkpoints/phase3/hardware",
                         lambda h: HardwareAwareDiffPool(in_channels=158, hidden_channels=h, dropout=0.0))}
CACHE = Path("checkpoints/r0_eval_cache")


def load(arm, s, dev):
    d, make = ARMS[arm]
    ck = torch.load(f"{d}-seed{s}/best_model.pt", map_location="cpu", weights_only=False)
    a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    m = make(a.get("hidden", 128))
    m.load_state_dict(ck["model_state"])
    return m.to(dev).eval()


@torch.no_grad()
def slowdowns(arm, coll, bs, dev):
    """{graph: {seed: top1}} for one arm/collection/batch, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{arm}_{coll.replace('/', '_')}_b{bs}.json"
    if cp.exists():
        return {g: {int(s): v for s, v in d.items()} for g, d in json.loads(cp.read_text()).items()}
    ds = TPUGraphDataset(f"{DATA}/{coll}/valid")
    groups = defaultdict(list)
    for flat, (fi, _) in enumerate(ds._index):
        groups[fi].append(flat)
    models = {s: load(arm, s, dev) for s in SEEDS}
    out = {}
    for fi, flats in groups.items():
        name = Path(ds._file_paths[fi]).stem
        if name not in CONV_GRAPHS | OTHER_GRAPHS:
            continue
        rt = np.asarray(ds._get_handle(fi)["config_runtime"], dtype=np.float64)[[ds._index[i][1] for i in flats]]
        preds = {s: [] for s in SEEDS}
        for st in range(0, len(flats), bs):
            batch = Batch.from_data_list([ds[i] for i in flats[st:st + bs]]).to(dev)
            for s, m in models.items():
                r = _forward(m, batch)
                preds[s].append((r[0] if isinstance(r, tuple) else r).float().cpu())
        out[name] = {s: float(rt[int(np.argmax(torch.cat(preds[s]).numpy()))] / rt.min()) for s in SEEDS}
        print(f"  {arm:<12} {coll:<11} b{bs:<3} {name[:36]:<38}" + " ".join(f"{out[name][s]:.3f}" for s in SEEDS), flush=True)
    cp.write_text(json.dumps(out))
    return out


def subset_means(raw, graphs):
    return [float(np.mean([raw[(c, g)][s] for (c, g) in raw if g in graphs])) for s in SEEDS]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for bs in BATCHES:
        res[bs] = {}
        for arm in ARMS:
            raw = {}
            for coll in COLLECTIONS:
                for g, sv in slowdowns(arm, coll, bs, dev).items():
                    raw[(coll, g)] = sv
            res[bs][arm] = {"conv": subset_means(raw, CONV_GRAPHS), "other": subset_means(raw, OTHER_GRAPHS),
                            "per_instance": {f"{c}::{g}": float(np.mean(list(v.values()))) for (c, g), v in raw.items()}}
    out, lines = {"eval_batches": BATCHES, "primary_batch": 32, "arms": {}, "contrasts": {}}, []
    lines += ["# R0 — CONV-subset Top-1 of structured arms on xla:random (exploratory, partially observed)", "",
              "Frozen: `r0_conv_top1_structure_declaration.md` + Amendment 1. Checkpoints: Phase 2c/3 xla:random, "
              "lr 1e-3, 5 seeds. Split: 8c frozen (CONV 3 × 2 collections, OTHER 4 × 2). "
              "**Primary = eval batch 32 (8c's); batch 8 = sensitivity.**", ""]
    for bs in BATCHES:
        lines += [f"## Eval batch {bs}{' (primary)' if bs == 32 else ' (sensitivity)'}", "",
                  "| arm | CONV Top-1 | OTHER Top-1 |", "|---|---|---|"]
        for arm in ARMS:
            c, o = res[bs][arm]["conv"], res[bs][arm]["other"]
            lines.append(f"| {arm} | {np.mean(c):.4f} ± {np.std(c, ddof=1):.4f} | {np.mean(o):.4f} ± {np.std(o, ddof=1):.4f} |")
        lines += ["", "| contrast (negative = closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |", "|---|---|---|---|---|---|---|"]
        for arm in ("hierarchical", "hardware"):
            for sub in ("conv", "other"):
                ps = paired_stats(res[bs][arm][sub], res[bs]["flat"][sub])
                sd = float(np.std(np.array(res[bs][arm][sub]) - np.array(res[bs]["flat"][sub]), ddof=1))
                ps["mde"] = min_detectable_effect(sd, 5) if sd > 0 else float("nan")
                ps["per_seed_delta"] = (np.array(res[bs][arm][sub]) - np.array(res[bs]["flat"][sub])).round(4).tolist()
                out["contrasts"][f"b{bs}:{arm}-flat:{sub}"] = ps
                lines.append(f"| {arm} − flat, {sub.upper()} | {ps['delta']:+.4f} | {ps['p_t']:.3g} | {ps['p_w']:.3g} | "
                             f"[{ps['ci'][0]:+.4f}, {ps['ci'][1]:+.4f}] | {ps['d']:.2f} | {ps['mde']:.4f} |")
        lines.append("")
        out["arms"][f"b{bs}"] = res[bs]
    lines += ["Reference (not a test; different harness, see declaration §3): 8c channel − baseline CONV Δ −0.3043, "
              "BH-p .0032 (`phase8c_results.md:12`, batch 32).", ""]
    Path("experiments/r0_conv_top1_results.json").write_text(json.dumps(out, indent=1))
    Path("experiments/r0_conv_top1_results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
