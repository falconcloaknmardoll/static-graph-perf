"""R5 evaluation (eval-only; experiments/r5_oracle_tuned_declaration.md, frozen 58cb565).

Primary: CONV Top-1, oracle-bucket@3e-5 − baseline@3e-5, paired by seed, n = 10, eval batch 32
(batch 8 = sensitivity). Placebo: the same on OTHER. Descriptive (§3.3): channel@3e-5 vs oracle@3e-5
and tuned recovery on 8e's 5 seeds; oracle@3e-5 vs oracle@1e-3 (8d); checkpoint epoch and val τ;
baseline on new vs 8e seeds. Verdict = declaration §6, from batch 32.

Baseline on 8e's seeds and channel@3e-5 come from 8e's caches; oracle@1e-3 from 8d's registered
cache (b32) and O19's (b8). Nothing registered is recomputed. New scores cache to checkpoints/r5_eval_cache/.
Top-1 = phase8c_evaluate's definition (runtime of the argmax-score config ÷ best runtime).
  python scripts/r5_evaluate.py   -> experiments/r5_results.{md,json}
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_mxu import MXUGraphDataset                               # noqa: E402
from dataset_oracle import OracleGraphDataset                         # noqa: E402
from model import BaselineGraphSAGE_Large                             # noqa: E402
from train import _forward                                            # noqa: E402
from phase8c_evaluate import COLLECTIONS, DATA, CONV_GRAPHS, OTHER_GRAPHS, paired_stats  # noqa: E402
from power_analysis import min_detectable_effect                      # noqa: E402

OLD_SEEDS = [42, 100, 123, 456, 789]            # 8e's
NEW_SEEDS = [7, 21, 314, 2718, 9001]            # declaration §2, fixed before any run
SEEDS = OLD_SEEDS + NEW_SEEDS
LR = "3e-5"
CACHE = Path("checkpoints/r5_eval_cache")
E8 = Path("checkpoints/phase8e/eval_cache")
D8 = {32: Path("checkpoints/phase8d/eval_cache"), 8: Path("checkpoints/o19_eval_cache/8d_b8")}
NAMES = CONV_GRAPHS | OTHER_GRAPHS


def ckpt(arm, s):
    if arm == "oracle":
        return Path(f"checkpoints/r5/xla_random/oracle-bucket-lr{LR}-seed{s}/best_model.pt")
    if s in OLD_SEEDS:
        return Path(f"checkpoints/phase8e/xla_random/mxu-baseline-lr{LR}-seed{s}/best_model.pt")
    return Path(f"checkpoints/r5/xla_random/mxu-baseline-lr{LR}-seed{s}/best_model.pt")


@torch.no_grad()
def score(arm, seeds, coll, bs, dev):
    """{graph: {seed: top1}} over NAMES on coll/valid for the given seeds. Cached per (arm, coll, bs)."""
    cpath = CACHE / f"{arm}_{coll.replace('/', '_')}_b{bs}.json"
    if cpath.exists():
        return {g: {int(s): v for s, v in d.items()} for g, d in json.loads(cpath.read_text()).items()}
    ds = (OracleGraphDataset(f"{DATA}/{coll}/valid") if arm == "oracle"
          else MXUGraphDataset(f"{DATA}/{coll}/valid", mxu_arm="baseline"))
    models = {}
    for s in seeds:
        ck = torch.load(ckpt(arm, s), map_location="cpu", weights_only=False)
        assert ck["mxu_arm"] == ("oracle-bucket" if arm == "oracle" else "baseline"), (s, ck["mxu_arm"])
        m = BaselineGraphSAGE_Large(in_channels=ck["in_dim"], dropout=0.0)
        m.load_state_dict(ck["model_state"])
        models[s] = m.to(dev).eval()
    groups = defaultdict(list)
    for flat, (fi, _) in enumerate(ds._index):
        groups[fi].append(flat)
    out = {}
    for fi, flats in groups.items():
        name = Path(ds._file_paths[fi]).stem
        if name not in NAMES:
            continue
        rt = np.asarray(ds._get_handle(fi)["config_runtime"], dtype=np.float64)[[ds._index[i][1] for i in flats]]
        preds = {s: [] for s in models}
        for start in range(0, len(flats), bs):
            batch = Batch.from_data_list([ds[i] for i in flats[start:start + bs]]).to(dev)
            for s, m in models.items():
                r = _forward(m, batch)
                preds[s].append((r[0] if isinstance(r, tuple) else r).cpu())
        out[name] = {s: float(rt[int(np.argmax(torch.cat(preds[s]).numpy()))] / rt.min()) for s in models}
        print(f"  {arm:<9} {coll:<11} b{bs:<3} {name[:38]:<40} "
              + "  ".join(f"{out[name][s]:.3f}" for s in seeds), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(out))
    return out


def read(p, key=None):
    d = json.loads(p.read_text())
    return {g: {int(s): v for s, v in (sv[key] if key else sv).items()} for g, sv in d.items()}


def per_seed(raw, names, seeds):
    """raw = {"coll::graph": {seed: x}} -> per-seed mean over instances whose graph is in names."""
    keys = [k for k in raw if k.split("::")[1] in names]
    return np.array([np.mean([raw[k][s] for k in keys]) for s in seeds])


def stats(a, b):
    r = paired_stats(list(a), list(b))
    d = np.asarray(a) - np.asarray(b)
    sd = float(d.std(ddof=1))
    r["mde"] = min_detectable_effect(sd, len(d)) if sd > 0 else float("nan")
    r["n"] = len(d)
    r["per_seed_delta"] = d.round(4).tolist()
    return r


HDR = ["| contrast (negative = first arm closer to optimal) | n | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |",
       "|---|---|---|---|---|---|---|---|"]
row = lambda lab, r: (f"| {lab} | {r['n']} | {r['delta']:+.4f} | {r['p_t']:.3g} | {r['p_w']:.3g} | "
                      f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['d']:.2f} | {r['mde']:.4f} |")
fmt = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    done = [s for s in SEEDS if ckpt("oracle", s).exists() and ckpt("baseline", s).exists()]
    missing = [s for s in SEEDS if s not in done]
    if missing:
        print(f"WARNING: seeds without both checkpoints {missing}; reported at n = {len(done)} (§7.2)", flush=True)
    new_done = [s for s in NEW_SEEDS if s in done]

    raw = {}
    for bs in (32, 8):
        raw[bs] = {"oracle": {}, "baseline": {}, "channel": {}, "oracle-1e-3": {}}
        for coll in COLLECTIONS:
            cs = coll.replace("/", "_")
            for g, sv in score("oracle", done, coll, bs, dev).items():
                raw[bs]["oracle"][f"{coll}::{g}"] = sv
            base = {g: dict(sv) for g, sv in read(E8 / f"baseline_lr{LR}_{cs}_b{bs}.json", "top1").items()}
            if new_done:
                for g, sv in score("baseline", new_done, coll, bs, dev).items():
                    base[g].update(sv)
            for g, sv in base.items():
                raw[bs]["baseline"][f"{coll}::{g}"] = sv
            for g, sv in read(E8 / f"channel_lr{LR}_{cs}_b{bs}.json", "top1").items():
                raw[bs]["channel"][f"{coll}::{g}"] = sv
            for g, sv in read(D8[bs] / f"oracle-bucket_{cs}.json").items():
                raw[bs]["oracle-1e-3"][f"{coll}::{g}"] = sv

    res = {"seeds": done, "missing": missing, "contrasts": {}, "means": {}, "per_seed": {}}
    L = ["# R5 — Oracle vs baseline, both at the tuned LR (3e-5), 10 seeds", "",
         "Frozen: `r5_oracle_tuned_declaration.md` (58cb565). CONV = 8c's frozen split (3 graphs × 2 xla "
         f"collections). **Primary = eval batch 32**; batch 8 = sensitivity. n = {len(done)}"
         + (f" (missing {missing}, §7.2)" if missing else "") + ". Baseline on 8e's seeds, channel@3e-5 and "
         "oracle@1e-3 are read from their registered caches.", ""]
    for bs in (32, 8):
        R = raw[bs]
        conv = {a: per_seed(R[a], CONV_GRAPHS, done) for a in ("oracle", "baseline")}
        other = {a: per_seed(R[a], OTHER_GRAPHS, done) for a in ("oracle", "baseline")}
        res["means"][f"b{bs}"] = {a: {"conv": float(conv[a].mean()), "other": float(other[a].mean())} for a in conv}
        res["per_seed"][f"b{bs}"] = {a: dict(zip(map(str, done), conv[a].round(4).tolist())) for a in conv}
        L += [f"## Eval batch {bs}{' (primary)' if bs == 32 else ' (sensitivity)'}", "",
              "| arm (lr 3e-5) | CONV Top-1 | OTHER Top-1 |", "|---|---|---|",
              f"| baseline | {fmt(conv['baseline'])} | {fmt(other['baseline'])} |",
              f"| oracle-bucket | {fmt(conv['oracle'])} | {fmt(other['oracle'])} |", ""] + HDR
        for key, sub, a, b in (("primary", "conv", conv["oracle"], conv["baseline"]),
                               ("placebo", "other", other["oracle"], other["baseline"])):
            r = stats(a, b)
            res["contrasts"][f"b{bs}:{key}"] = r
            L.append(row(f"{key}: oracle − baseline, {sub.upper()}", r))
        pr, pl = res["contrasts"][f"b{bs}:primary"], res["contrasts"][f"b{bs}:placebo"]
        if pl["delta"] <= pr["delta"]:
            L.append("\nFLAG (§3.2): OTHER gain ≥ CONV gain → not conv-specific (8d's rule).")
        ceiling = float(conv["baseline"].mean() - 1.0)
        res["contrasts"][f"b{bs}:primary"]["ceiling"] = ceiling
        L += ["", f"Satisfiability (§7.4): realised ceiling {ceiling:.4f} vs realised MDE {pr['mde']:.4f}"
              + (" → **unsatisfiable after the fact; state beside the verdict.**" if pr["mde"] > ceiling else " → satisfiable."),
              "", "Per seed (CONV Top-1):", "", "| seed | baseline | oracle | Δ |", "|---|---|---|---|"]
        for i, s in enumerate(done):
            L.append(f"| {s}{'' if s in OLD_SEEDS else ' (new)'} | {conv['baseline'][i]:.4f} | "
                     f"{conv['oracle'][i]:.4f} | {conv['oracle'][i] - conv['baseline'][i]:+.4f} |")

        # §3.3 descriptive, on 8e's seeds that finished
        old = [s for s in OLD_SEEDS if s in done]
        c = {a: per_seed(R[a], CONV_GRAPHS, old) for a in ("baseline", "channel", "oracle", "oracle-1e-3")}
        rc = stats(c["channel"], c["oracle"])
        ro = stats(c["oracle"], c["oracle-1e-3"])
        rec = float((c["baseline"].mean() - c["channel"].mean()) / (c["baseline"].mean() - c["oracle"].mean()))
        bn = per_seed(R["baseline"], CONV_GRAPHS, new_done) if new_done else np.array([np.nan])
        res["descriptive"] = res.get("descriptive", {})
        res["descriptive"][f"b{bs}"] = {"channel_minus_oracle": rc, "oracle3e5_minus_oracle1e3": ro,
                                        "tuned_recovery": rec, "baseline_old_seeds": float(c["baseline"].mean()),
                                        "baseline_new_seeds": float(np.mean(bn))}
        L += ["", f"Descriptive (§3.3, 8e's seeds, n = {len(old)}; no verdict):", "", *HDR,
              row("channel@3e-5 − oracle@3e-5, CONV", rc),
              row("oracle@3e-5 − oracle@1e-3 (8d; also GPU vs CPU), CONV", ro),
              "", f"- Tuned recovery (baseline − channel) / (baseline − oracle): {rec:.3f}",
              f"- Baseline CONV, 8e seeds {c['baseline'].mean():.4f} vs new seeds {np.mean(bn):.4f}", ""]

    ep = {s: torch.load(ckpt("oracle", s), map_location="cpu", weights_only=False) for s in done}
    bp = {s: torch.load(ckpt("baseline", s), map_location="cpu", weights_only=False) for s in done}
    res["checkpoints"] = {a: {str(s): {"epoch": int(d[s]["epoch"]), "val_tau": float(d[s]["val_tau"])} for s in d}
                          for a, d in (("oracle", ep), ("baseline", bp))}
    L += ["## Checkpoints (selected by overall val τ; the oracle's τ is inflated by its leaked input, O23)", "",
          "| seed | baseline epoch | baseline val τ | oracle epoch | oracle val τ |", "|---|---|---|---|---|"]
    L += [f"| {s} | {bp[s]['epoch']} | {bp[s]['val_tau']:.4f} | {ep[s]['epoch']} | {ep[s]['val_tau']:.4f} |" for s in done]

    p, p8 = res["contrasts"]["b32:primary"], res["contrasts"]["b8:primary"]
    verdict = ("headroom after tuning: the oracle beats the tuned baseline" if p["delta"] < 0 and p["p_t"] < .05 else
               "no headroom detected after tuning" if p["delta"] < 0 else
               "a tuned oracle does not beat the tuned baseline")
    res["verdict"] = verdict
    L += ["", "## Verdict (declaration §6, batch 32)", "",
          f"- Δ {p['delta']:+.4f}, 95% CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}], p = {p['p_t']:.3g}, "
          f"MDE {p['mde']:.4f}, n = {p['n']} → **{verdict}**.",
          f"- Batch 8: Δ {p8['delta']:+.4f}, p = {p8['p_t']:.3g}"
          + (" — **sign differs from batch 32; disclose in the same sentence (§6).**"
             if np.sign(p8["delta"]) != np.sign(p["delta"]) else ".")]
    out = Path("experiments/r5_results.md")
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L), f"\nWrote {out} and {out.with_suffix('.json')}", sep="\n")


if __name__ == "__main__":
    main()
