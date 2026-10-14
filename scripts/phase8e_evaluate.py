"""Phase 8e evaluation (experiments/phase8e_lr_robustness_preregistration.md §4-§7 + Amendment 1).

Q1 (primary, two-sided): CONV Top-1, channel@LR*_ch - baseline@LR*_b, paired by seed.
Q2 (secondary): CONV Top-1, channel@1e-3 - baseline@LR*_b. Holm over {Q1, Q2}; Q1's verdict uses its
unadjusted p (§3). LR* from scripts/phase8e_lr_star.py (overall val tau, never CONV Top-1).
Top-1 = phase8c_evaluate's definition, frozen 8c split, both xla valid sets, eval batch 32 (Amendment 1);
batch 8 reported beside it. lr-1e-3 cells: registered 8c caches at batch 32 (re-scored here as a
reproduction gate) and O19's caches at batch 8 (pending until queue o has run).
Report-either-way items (§5): overall val tau per arm x LR, OTHER placebo, per-instance CONV table,
convergence, CONV Spearman(pred, runtime) at LR*. Figure: CONV Top-1 and overall tau vs LR.

Each eval batch is built once per (arm, collection, batch size) and run through every model of that
arm, as in phase8c_evaluate.py. Caches: checkpoints/phase8e/eval_cache/.
  python scripts/phase8e_evaluate.py   -> experiments/phase8e_results.{md,json}, experiments/figures/phase8e_lr_curves.png
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_mxu import MXUGraphDataset                               # noqa: E402
from model import BaselineGraphSAGE_Large                             # noqa: E402
from train import _forward                                            # noqa: E402
from phase8c_evaluate import (SEEDS, COLLECTIONS, DATA, CONV_GRAPHS, OTHER_GRAPHS,  # noqa: E402
                              paired_stats)
from power_analysis import min_detectable_effect                      # noqa: E402

ARMS = ["baseline", "channel"]
NEW_LRS = ["3e-5", "1e-4", "3e-4"]
ALL_LRS = ["3e-5", "1e-4", "3e-4", "1e-3"]
CACHE = Path("checkpoints/phase8e/eval_cache")
REG_8C = Path("checkpoints/phase8c/eval_cache")                       # registered lr-1e-3, batch 32
O19_B8 = Path("checkpoints/o19_eval_cache/8c_b8")                     # lr-1e-3, batch 8 (queue o)
OUT = Path("experiments/phase8e_results.md")
FIG = Path("experiments/figures/phase8e_lr_curves.png")


def ckpt(arm, lr, s):
    if lr == "1e-3":
        return Path(f"checkpoints/phase8c/xla_random/mxu-{arm}-seed{s}/best_model.pt")
    return Path(f"checkpoints/phase8e/xla_random/mxu-{arm}-lr{lr}-seed{s}/best_model.pt")


def lr_star(arm):
    out = subprocess.run([sys.executable, "scripts/phase8e_lr_star.py", "--arm", arm, "--print-lr-star"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def load(arm, lr, s, dev):
    ck = torch.load(ckpt(arm, lr, s), map_location="cpu", weights_only=False)
    assert ck["mxu_arm"] == arm, (arm, lr, s, ck["mxu_arm"])
    m = BaselineGraphSAGE_Large(in_channels=ck["in_dim"], dropout=0.0)
    m.load_state_dict(ck["model_state"])
    return m.to(dev).eval(), ck


@torch.no_grad()
def score(arm, coll, bs, lrs, dev):
    """{lr: {graph: {"top1": {seed: x}, "rho": {seed: x}}}} for one (arm, collection, batch size).
    Each batch is built once and run through all len(lrs) x 5 models."""
    CACHE.mkdir(parents=True, exist_ok=True)
    c = coll.replace("/", "_")
    paths = {lr: CACHE / f"{arm}_lr{lr}_{c}_b{bs}.json" for lr in lrs}
    todo = [lr for lr in lrs if not paths[lr].exists()]
    if todo:
        ds = MXUGraphDataset(f"{DATA}/{coll}/valid", mxu_arm=arm)
        models = {(lr, s): load(arm, lr, s, dev)[0] for lr in todo for s in SEEDS}
        groups = defaultdict(list)
        for flat, (fi, _) in enumerate(ds._index):
            groups[fi].append(flat)
        out = {lr: {} for lr in todo}
        for fi, flats in groups.items():
            name = Path(ds._file_paths[fi]).stem
            if name not in CONV_GRAPHS | OTHER_GRAPHS:
                continue
            rt = np.asarray(ds._get_handle(fi)["config_runtime"], dtype=np.float64)[[ds._index[i][1] for i in flats]]
            preds = defaultdict(list)
            for start in range(0, len(flats), bs):
                batch = Batch.from_data_list([ds[i] for i in flats[start:start + bs]]).to(dev)
                for key, m in models.items():
                    r = _forward(m, batch)
                    preds[key].append((r[0] if isinstance(r, tuple) else r).cpu())
            for lr in todo:
                out[lr][name] = {"top1": {}, "rho": {}}
                for s in SEEDS:
                    p = torch.cat(preds[(lr, s)]).numpy()
                    out[lr][name]["top1"][s] = float(rt[int(np.argmax(p))] / rt.min())
                    out[lr][name]["rho"][s] = float(spearmanr(p, rt).correlation)
            print(f"  {arm:<8} b{bs:<2} {coll:<11} {name[:36]:<38} "
                  + "  ".join(f"{lr}:{np.mean(list(out[lr][name]['top1'].values())):.3f}" for lr in todo), flush=True)
        for lr in todo:
            paths[lr].write_text(json.dumps(out[lr]))
    res = {}
    for lr in lrs:
        d = json.loads(paths[lr].read_text())
        res[lr] = {g: {k: {int(s): v for s, v in sv.items()} for k, sv in gd.items()} for g, gd in d.items()}
    return res


def top1_only(path):
    return {g: {int(s): v for s, v in d.items()} for g, d in json.loads(path.read_text()).items()}


def subset(cells, names):
    """cells: {coll: {graph: {seed: top1}}} -> per-seed mean over instances in `names`."""
    return [float(np.mean([cells[c][g][s] for c in cells for g in cells[c] if g in names])) for s in SEEDS]


def stats(a, b):
    r = paired_stats(a, b)
    d = np.array(a) - np.array(b)
    r["mde"] = float(min_detectable_effect(float(d.std(ddof=1)), len(SEEDS)))
    r["per_seed_delta"] = d.round(4).tolist()
    return r


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    star = {a: lr_star(a) for a in ARMS}
    print(f"device={dev}  LR*={star}", flush=True)

    top1 = {32: defaultdict(dict), 8: defaultdict(dict)}   # bs -> (arm, lr) -> coll -> graph -> seed
    rho = defaultdict(dict)                                 # (arm, lr) -> coll -> graph -> seed
    gate = {}
    for arm in ARMS:
        for coll in COLLECTIONS:
            c = coll.replace("/", "_")
            r32 = score(arm, coll, 32, ALL_LRS, dev)        # 1e-3 included = reproduction gate
            for lr in NEW_LRS:
                top1[32][(arm, lr)][coll] = {g: v["top1"] for g, v in r32[lr].items()}
                rho[(arm, lr)][coll] = {g: v["rho"] for g, v in r32[lr].items()}
            reg = top1_only(REG_8C / f"{arm}_{c}.json")
            top1[32][(arm, "1e-3")][coll] = reg                  # registered numbers stay primary
            diffs = [abs(r32["1e-3"][g]["top1"][s] - reg[g][s]) for g in reg for s in SEEDS]
            gate[f"{arm} {coll}"] = {"cells": len(diffs), "differing": int(sum(x > 1e-9 for x in diffs)),
                                     "max_abs_diff": float(max(diffs))}
            r8 = score(arm, coll, 8, [star[arm]], dev)
            top1[8][(arm, star[arm])][coll] = {g: v["top1"] for g, v in r8[star[arm]].items()}
            o19 = O19_B8 / f"{arm}_{c}.json"
            if o19.exists():
                top1[8][(arm, "1e-3")][coll] = top1_only(o19)

    def cell(bs, arm, lr, names):
        t = top1[bs].get((arm, lr))
        return subset(t, names) if t and len(t) == len(COLLECTIONS) else None

    res = {"lr_star": star, "reproduction_gate_8c_b32": gate, "contrasts": {}, "overall_val_tau": {},
           "best_epoch": {}, "conv_spearman_at_lr_star": {}, "per_instance_conv_at_lr_star": {}}
    L = ["# Phase 8e — results (LR robustness of the 8c channel effect)", "",
         f"Pre-registration: `phase8e_lr_robustness_preregistration.md` (frozen 44267c7) + Amendment 1. "
         f"LR* (overall xla:random val τ, boundary rule applied): baseline **{star['baseline']}**, channel **{star['channel']}**. "
         "CONV Top-1 at eval batch 32 (primary, Amendment 1); batch 8 beside it.", ""]

    # overall val tau + convergence (from the checkpoints themselves)
    L += ["## Overall xla:random val τ by arm × LR (§5 item 1; checkpoint `val_tau`)", "",
          "| arm | " + " | ".join(ALL_LRS) + " |", "|---|" + "---|" * len(ALL_LRS)]
    for arm in ARMS:
        row = []
        for lr in ALL_LRS:
            cks = [torch.load(ckpt(arm, lr, s), map_location="cpu", weights_only=False) for s in SEEDS]
            v = [float(c["val_tau"]) for c in cks]
            res["overall_val_tau"][f"{arm}@{lr}"] = v
            res["best_epoch"][f"{arm}@{lr}"] = [int(c["epoch"]) for c in cks]
            row.append(f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")
        L.append(f"| {arm} | " + " | ".join(row) + " |")

    # Q1, Q2 (+ placebo), batch 32 primary and batch 8 sensitivity
    q = {"Q1": ("channel", star["channel"], "baseline", star["baseline"]),
         "Q2": ("channel", "1e-3", "baseline", star["baseline"])}
    for bs in (32, 8):
        for k, (a1, l1, a2, l2) in q.items():
            for sub, names in (("conv", CONV_GRAPHS), ("other", OTHER_GRAPHS)):
                x, y = cell(bs, a1, l1, names), cell(bs, a2, l2, names)
                res["contrasts"][f"b{bs}:{k}:{sub}"] = stats(x, y) if x and y else "pending (O19 batch-8 cache not yet computed)"
    p1, p2 = res["contrasts"]["b32:Q1:conv"]["p_t"], res["contrasts"]["b32:Q2:conv"]["p_t"]
    first, second = min(1.0, 2 * min(p1, p2)), min(1.0, max(2 * min(p1, p2), max(p1, p2)))  # Holm, m = 2
    res["contrasts"]["b32:Q1:conv"]["p_holm"] = first if p1 <= p2 else second
    res["contrasts"]["b32:Q2:conv"]["p_holm"] = first if p2 < p1 else second

    def verdict(r):
        if r["delta"] < 0 and r["p_t"] < .05:
            return "8c effect survives symmetric LR tuning"
        if r["delta"] < 0:
            return "Not shown robust to LR tuning"
        return "The 8c advantage is attributable to LR / training dynamics"

    base_conv = float(np.mean(cell(32, "baseline", star["baseline"], CONV_GRAPHS)))
    q1 = res["contrasts"]["b32:Q1:conv"]
    res["verdict_Q1"] = verdict(q1)
    res["verdict_Q2"] = verdict(res["contrasts"]["b32:Q2:conv"])
    conv_ep = {a: sum(e >= 46 for e in res["best_epoch"][f"{a}@{star[a]}"]) for a in ARMS}
    budget = any(n >= 3 for n in conv_ep.values())
    L += ["", "## Registered tests (§3, §7) — CONV Top-1, paired by seed, eval batch 32", "",
          "| test | contrast | Δ | 95% CI | t-p | Holm-p | Wilcoxon-p | d | MDE |", "|---|---|---|---|---|---|---|---|---|"]
    for k, (a1, l1, a2, l2) in q.items():
        r = res["contrasts"][f"b32:{k}:conv"]
        L.append(f"| {k} | {a1}@{l1} − {a2}@{l2} | {r['delta']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                 f"{r['p_t']:.4f} | {r['p_holm']:.4f} | {r['p_w']:.3f} | {r['d']:+.2f} | {r['mde']:.4f} |")
    L += ["", f"**Q1 verdict (§7, unadjusted p): {res['verdict_Q1']}.** Q2 (same wording): {res['verdict_Q2']}.",
          f"- Satisfiability boundary (§6): tuned baseline CONV Top-1 = {base_conv:.4f} "
          f"({'above' if base_conv > 1.19 else 'BELOW'} the ≈1.16–1.19 boundary).",
          f"- Convergence (§5 item 4): runs at LR* peaking at epoch ≥ 46: baseline {conv_ep['baseline']}/5, "
          f"channel {conv_ep['channel']}/5 → " + ('verdict carries "within a 50-epoch budget"' if budget else "no budget qualifier") + ".",
          f"- LR* = {star['baseline']} / {star['channel']} is the lower edge of the enlarged grid; §4 allows no further step. Disclose."]

    L += ["", "## Reported either way (§5)", "", "| item | subset | b32 Δ | b32 p | b8 Δ | b8 p |", "|---|---|---|---|---|---|"]
    for k in q:
        for sub in ("conv", "other"):
            r32, r8 = res["contrasts"][f"b32:{k}:{sub}"], res["contrasts"][f"b8:{k}:{sub}"]
            f = lambda r: (f"{r['delta']:+.4f}", f"{r['p_t']:.4f}") if isinstance(r, dict) else ("pending O19", "")
            label = {"conv": "CONV", "other": "OTHER (placebo)"}[sub]
            L.append(f"| {k} | {label} | {f(r32)[0]} | {f(r32)[1]} | {f(r8)[0]} | {f(r8)[1]} |")

    L += ["", "## Per-instance CONV Top-1 at LR* (§5 item 3; mean over seeds, batch 32)", "",
          "| collection | graph | baseline@" + star["baseline"] + " | channel@" + star["channel"] + " |", "|---|---|---|---|"]
    for coll in COLLECTIONS:
        for g in sorted(CONV_GRAPHS):
            b = np.mean(list(top1[32][("baseline", star["baseline"])][coll][g].values()))
            ch = np.mean(list(top1[32][("channel", star["channel"])][coll][g].values()))
            res["per_instance_conv_at_lr_star"][f"{coll}::{g}"] = {"baseline": float(b), "channel": float(ch)}
            L.append(f"| {coll} | {g} | {b:.4f} | {ch:.4f} |")

    L += ["", "## CONV Spearman(pred, runtime) at LR* (§5 item 5; mean over graphs × seeds; good ranking ≪ 0)", "",
          "| arm | xla/default | xla/random |", "|---|---|---|"]
    for arm in ARMS:
        vals = {coll: float(np.nanmean([v for g in CONV_GRAPHS for v in rho[(arm, star[arm])][coll][g].values()]))
                for coll in COLLECTIONS}
        res["conv_spearman_at_lr_star"][arm] = vals
        L.append(f"| {arm}@{star[arm]} | {vals['xla/default']:+.3f} | {vals['xla/random']:+.3f} |")

    L += ["", "## Reproduction gate: 8c lr-1e-3 cells re-scored at batch 32 vs the registered caches", ""]
    L += [f"- {k}: {v['differing']}/{v['cells']} graph×seed cells differ, max |Δ| {v['max_abs_diff']:.2e}" for k, v in gate.items()]

    # Figure (§5 FIG): CONV Top-1 and overall tau vs LR, both arms, seeds as dots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    xs = np.arange(len(ALL_LRS))
    for i, arm in enumerate(ARMS):
        col = ["#4c72b0", "#dd8452"][i]
        conv = [cell(32, arm, lr, CONV_GRAPHS) for lr in ALL_LRS]
        tau = [res["overall_val_tau"][f"{arm}@{lr}"] for lr in ALL_LRS]
        for ax, ys in ((axs[0], conv), (axs[1], tau)):
            for x, y in zip(xs, ys):
                ax.scatter([x + (i - .5) * .12] * len(y), y, s=10, color=col, alpha=.6)
            ax.plot(xs + (i - .5) * .12, [np.mean(y) for y in ys], color=col, label=arm)
    for ax, t in ((axs[0], "CONV Top-1 slowdown (batch 32)"), (axs[1], "overall val τ (xla:random)")):
        ax.set_xticks(xs, ALL_LRS); ax.set_xlabel("learning rate"); ax.set_ylabel(t)
    axs[0].legend(frameon=False)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=200)
    L += ["", f"Figure: `{FIG}`."]

    OUT.write_text("\n".join(L) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
