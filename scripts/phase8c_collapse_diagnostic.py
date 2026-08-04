"""
phase8c_collapse_diagnostic.py — Confound 1b for the Phase-8c result.

The 8c crux ([[project-phase8c-result]]): MXU features are config-INVARIANT
(same block appended to every config of a graph), yet they change WITHIN-graph
config ranking. The only mechanism that can do that is changed training dynamics.
The proposed signature is a "prediction collapse": the BASELINE model, on conv
graphs, emits near-CONSTANT scores across a graph's configs (so argmax is
essentially arbitrary → bad Top-1), and the MXU features BREAK that collapse
(scores spread out → the ranking becomes meaningful).

This script tests that signature directly, using ONLY the existing trained
checkpoints — NO new training, runs on CPU. It is a DIAGNOSTIC, not a
pre-registered hypothesis test.

Per (arm, collection, graph, seed) it computes:
  • disp   = std(pred) / (|mean(pred)| + eps)   — config-wise prediction spread
  • rho    = Spearman(pred, runtime)            — rankability (want << 0: higher
             score ⇒ lower runtime, since Top-1 = argmax(pred))
and aggregates CONV vs OTHER, baseline vs feature arms.

Usage:  python scripts/phase8c_collapse_diagnostic.py [--arms baseline channel]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (dataset_mxu etc.)

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Batch

from dataset_mxu import MXUGraphDataset, ARM_COLUMNS
from train import _forward
from phase8c_evaluate import (
    SEEDS, COLLECTIONS, DATA, CONV_GRAPHS, OTHER_GRAPHS, load_arm_models,
)

OUT = Path("experiments/phase8c_collapse_diagnostic.json")


@torch.no_grad()
def per_config_preds(arm: str, coll: str, device, batch_size: int = 32):
    """{graph_name: {seed: (pred[np], runtime[np])}} for one (arm, collection)."""
    ds = MXUGraphDataset(f"{DATA}/{coll}/valid", mxu_arm=arm)
    models = load_arm_models(arm, device)

    groups: dict[int, list[int]] = defaultdict(list)
    for flat, (fi, _) in enumerate(ds._index):
        groups[fi].append(flat)

    out: dict[str, dict[int, tuple]] = {}
    for fi, flats in groups.items():
        name = Path(ds._file_paths[fi]).stem
        if name not in CONV_GRAPHS | OTHER_GRAPHS:
            continue
        npz = ds._get_handle(fi)
        cfg_idx = [ds._index[i][1] for i in flats]
        runtime = np.asarray(npz["config_runtime"], dtype=np.float64)[cfg_idx]

        preds = {s: [] for s in SEEDS}
        for start in range(0, len(flats), batch_size):
            chunk = flats[start:start + batch_size]
            batch = Batch.from_data_list([ds[i] for i in chunk]).to(device)
            for s, m in models.items():
                r = _forward(m, batch)
                preds[s].append((r[0] if isinstance(r, tuple) else r).cpu())

        out[name] = {s: (torch.cat(preds[s]).numpy().ravel(), runtime) for s in SEEDS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline", "channel", "spatial", "full"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}   arms={args.arms}\n")
    eps = 1e-12

    # rows: (arm, coll, subset, graph, seed, n_cfg, disp, rho)
    rows = []
    for arm in args.arms:
        for coll in COLLECTIONS:
            data = per_config_preds(arm, coll, device)
            for name, byseed in data.items():
                subset = "CONV" if name in CONV_GRAPHS else "OTHER"
                for s, (p, rt) in byseed.items():
                    disp = float(np.std(p) / (abs(np.mean(p)) + eps))
                    rho = float(spearmanr(p, rt).correlation) if len(p) > 2 else float("nan")
                    rows.append((arm, coll, subset, name, s, len(p), disp, rho))

    OUT.write_text(json.dumps(rows, indent=1))

    # ── aggregate: mean over graphs×seeds, per (arm, subset), per collection ──
    def agg(coll, subset, arm, col):
        v = [r[col] for r in rows if r[1] == coll and r[2] == subset and r[0] == arm]
        v = [x for x in v if not np.isnan(x)]
        return (np.mean(v), np.std(v)) if v else (float("nan"), float("nan"))

    for coll in COLLECTIONS:
        print(f"\n═══ collection {coll} ═══")
        print(f"{'arm':<10} {'subset':<6} {'disp std/|mean|':>18} {'Spearman(pred,rt)':>20}")
        for subset in ("CONV", "OTHER"):
            for arm in args.arms:
                dm, ds_ = agg(coll, subset, arm, 6)
                rm, rs = agg(coll, subset, arm, 7)
                print(f"{arm:<10} {subset:<6} {dm:>10.4f}±{ds_:<6.4f} {rm:>+11.4f}±{rs:<6.4f}")

    print(f"\nwrote {OUT}")
    print("\nCollapse signature CONFIRMED if, on CONV:  baseline disp ≪ channel disp")
    print("and baseline rho ≈ 0 while channel rho ≪ 0 (features restore rankability).")


if __name__ == "__main__":
    main()
