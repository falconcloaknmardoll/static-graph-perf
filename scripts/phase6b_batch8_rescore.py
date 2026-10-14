"""O21 — Phase 6b per-graph tau re-scored at eval batch 8 (eval-only; experiments/o21_6b_batch8_declaration.md).

Same checkpoints (xla:default, flat + hardware, lr 1e-3, 5 seeds), same per-graph tau code
(scripts/phase6b_confound_checks.per_graph_tau), eval batch 8 instead of its default 32. Writes new
files; the batch-32 originals (experiments/phase6b_confound_checks.{md,json}) are never rewritten.
Reports, at both batches: the per-graph table and inception_v3's share (O3), the 3-way BH over the
Phase 6b family (flat-vs-hardware recomputed at the batch in question), and the tightening curve (FIG-M2)
from the flat arm. Gate: each seed's mean over the 7 graphs must equal that checkpoint's stored batch-8
val tau (they are the same computation).
  python scripts/phase6b_batch8_rescore.py   -> experiments/phase6b_confound_checks_b8.{md,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6b_confound_checks import (SEEDS, VALID_DIR, CKPT, TPUGraphDataset, load_model,  # noqa: E402
                                     per_graph_tau, bh_correct)
from power_analysis import tightening_curve                           # noqa: E402

REG = Path("experiments/phase6b_confound_checks.json")                # batch 32, registered
OUT = Path("experiments/phase6b_confound_checks_b8.md")
ARMS = ["wide-baseline", "hardware"]


def summarise(per_graph):
    """Per-graph mean delta, inception_v3 share, per-seed means, flat-vs-hardware paired t."""
    graphs = list(per_graph["wide-baseline"])
    delta = {g: float(np.mean(per_graph["hardware"][g]) - np.mean(per_graph["wide-baseline"][g])) for g in graphs}
    others = [d for g, d in delta.items() if g != "inception_v3_batch_128_train"]
    seed_means = {a: [float(np.mean([per_graph[a][g][i] for g in graphs])) for i in range(len(SEEDS))] for a in ARMS}
    t, p = sp_stats.ttest_rel(seed_means["hardware"], seed_means["wide-baseline"])
    return {"delta": delta, "mean_delta_all": float(np.mean(list(delta.values()))),
            "mean_delta_without_inception": float(np.mean(others)),
            "inception_delta": delta["inception_v3_batch_128_train"], "seed_means": seed_means,
            "flat_vs_hardware": {"t": float(t), "p": float(p)},
            "bh_3way": dict(zip(["flat_vs_hier", "hier_vs_hardware", "flat_vs_hardware"],
                                bh_correct([0.0019, 0.0299, float(p)])))}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = TPUGraphDataset(data_dir=VALID_DIR)
    pg8 = {a: {} for a in ARMS}
    gate = {}
    for a in ARMS:
        stored = []
        for s in SEEDS:
            m = load_model(a, s, dev)
            for g, t in per_graph_tau(m, val_ds, dev, eval_batch_size=8).items():
                pg8[a].setdefault(g, []).append(t)
            stored.append(float(torch.load(CKPT / f"{a}-seed{s}" / "best_model.pt", map_location="cpu",
                                           weights_only=False)["val_tau"]))
            del m
            print(f"  {a} seed {s} done", flush=True)
        recomputed = [float(np.mean([pg8[a][g][i] for g in pg8[a]])) for i in range(len(SEEDS))]
        gate[a] = {"stored_val_tau": stored, "recomputed_b8": recomputed,
                   "max_abs_diff": float(max(abs(x - y) for x, y in zip(stored, recomputed)))}

    reg = json.loads(REG.read_text())
    s32, s8 = summarise(reg["per_graph_tau"]), summarise(pg8)
    tc = {b: {r["n_graphs"]: r for r in tightening_curve(pg["wide-baseline"], list(range(2, 21)), n_bootstrap=2000)}
          for b, pg in ((32, reg["per_graph_tau"]), (8, pg8))}

    L = ["# Phase 6b per-graph τ at eval batch 8 vs 32 (O21; exploratory sensitivity)", "",
         "Declared in `experiments/o21_6b_batch8_declaration.md`. Batch 32 = the original "
         "`phase6b_confound_checks.json` (unchanged); batch 8 = this re-score. xla:default, lr 1e-3, 5 seeds.", "",
         "## Gate: per-seed mean over the 7 graphs vs the checkpoint's stored batch-8 val τ", ""]
    L += [f"- {a}: max |Δ| {v['max_abs_diff']:.2e}" for a, v in gate.items()]
    L += ["", "## Per-graph Δτ (hardware − flat, mean over seeds)", "", "| graph | b32 | b8 |", "|---|---|---|"]
    for g in s32["delta"]:
        L.append(f"| {g} | {s32['delta'][g]:+.3f} | {s8['delta'][g]:+.3f} |")
    L += ["", "| quantity | b32 | b8 |", "|---|---|---|",
          f"| inception_v3 Δ | {s32['inception_delta']:+.3f} | {s8['inception_delta']:+.3f} |",
          f"| mean of 7 per-graph Δ | {s32['mean_delta_all']:+.3f} | {s8['mean_delta_all']:+.3f} |",
          f"| mean of the other 6 | {s32['mean_delta_without_inception']:+.3f} | {s8['mean_delta_without_inception']:+.3f} |",
          f"| flat-vs-hardware paired p | {s32['flat_vs_hardware']['p']:.2e} | {s8['flat_vs_hardware']['p']:.2e} |",
          f"| BH-adjusted p (flat-vs-hier / hier-vs-hw / flat-vs-hw) | "
          + " / ".join(f"{v:.4f}" for v in s32["bh_3way"].values()) + " | "
          + " / ".join(f"{v:.4f}" for v in s8["bh_3way"].values()) + " |",
          f"| tightening, flat: median SD at n = 7 (p05–p95) | {tc[32][7]['std_median']:.4f} "
          f"({tc[32][7]['std_p05']:.4f}–{tc[32][7]['std_p95']:.4f}) | {tc[8][7]['std_median']:.4f} "
          f"({tc[8][7]['std_p05']:.4f}–{tc[8][7]['std_p95']:.4f}) |",
          f"| tightening, flat: median SD at n = 20 | {tc[32][20]['std_median']:.4f} | {tc[8][20]['std_median']:.4f} |"]
    OUT.write_text("\n".join(L) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps(
        {"gate": gate, "per_graph_tau_b8": pg8, "summary_b32": s32, "summary_b8": s8,
         "tightening": {str(b): {str(k): v for k, v in t.items()} for b, t in tc.items()}}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
