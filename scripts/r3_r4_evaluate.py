"""R3 and R4 evaluation (eval-only; frozen declarations, scored as written).

R3 — experiments/r3_scramble_gpu_replication_declaration.md: the scramble arm re-trained on GPU
     (checkpoints/r3/xla_random). Contrasts at eval batch 32 (primary, as P4) and 8:
       1. R3-P4: channel (8c, GPU) − scramble-GPU, CONV
       2. scramble-GPU − baseline (8c, GPU), CONV and OTHER placebo
       3. scramble-GPU − scramble-CPU (8c), CONV and OTHER (descriptive hardware offset)
     8c comparators come from the registered caches (batch 32) and O19's caches (batch 8); never recomputed.
R4 — experiments/r4_heldout_conv_declaration.md + Amendment 1: baseline and channel at their 8e LR*
     (checkpoints/r4/xla_random_r4), trained without the 11 held-out conv graphs. Primary
     (exploratory): mean Top-1 over the 19 held-out instances, channel − baseline, paired by seed.
     Placebo: 8c's frozen OTHER validation subset. Per-family, per-instance, leave-one-family-out.

Top-1 = phase8c_evaluate's definition (runtime of the argmax-score config ÷ best runtime, every
config of the instance). Each eval batch is built once and run through every model of the arm.
Caches: checkpoints/r3r4_eval_cache/.
  python scripts/r3_r4_evaluate.py --which r3|r4   -> experiments/r{3,4}_results.{md,json}
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_mxu import MXUGraphDataset                               # noqa: E402
from model import BaselineGraphSAGE_Large                             # noqa: E402
from train import _forward                                            # noqa: E402
from phase8c_evaluate import (SEEDS, COLLECTIONS, DATA, CONV_GRAPHS, OTHER_GRAPHS,  # noqa: E402
                              paired_stats)
from power_analysis import min_detectable_effect                      # noqa: E402

CACHE = Path("checkpoints/r3r4_eval_cache")
REG_8C = Path("checkpoints/phase8c/eval_cache")       # registered 8c caches, batch 32 (read-only)
O19_B8 = Path("checkpoints/o19_eval_cache/8c_b8")     # O19's 8c caches, batch 8 (read-only)
R4_FAMILIES = {
    "alexnet": ["alexnet_train_batch_32"],
    "maskrcnn": ["mlperf_maskrcnn_1_shard_batch_4", "mlperf_maskrcnn_batch_2", "mlperf_maskrcnn_batch_4"],
    "ssd": ["mlperf_ssd_1_shard_batch_8_fast_epoch", "mlperf_ssd_2_shard_batch_8_fast_epoch"],
    "mnasnet": ["mnasnet_a1_batch_128"],
    "retinanet": ["retinanet.2x2.fp32", "retinanet.4x4.bf16.performance", "retinanet.4x4.fp32"],
    "xception": ["xception_imagenet"],
}
R4_HELD_OUT = {g for gs in R4_FAMILIES.values() for g in gs}
R4_LR = "3e-5"   # both arms' final 8e LR* (phase8e_results.md); asserted against phase8e_lr_star.py below


def load_models(paths, arm, dev):
    models = {}
    for s, p in paths.items():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert ck["mxu_arm"] == arm, (p, ck["mxu_arm"])
        m = BaselineGraphSAGE_Large(in_channels=ck["in_dim"], dropout=0.0)
        m.load_state_dict(ck["model_state"])
        models[s] = m.to(dev).eval()
    return models


@torch.no_grad()
def score(tag, paths, arm, data_dir, names, bs, dev):
    """{graph: {seed: top1}} for every graph in `names` found in data_dir. Cached per (tag, dir, bs)."""
    cpath = CACHE / f"{tag}_{data_dir.replace('/', '_')}_b{bs}.json"
    if cpath.exists():
        return {g: {int(s): v for s, v in d.items()} for g, d in json.loads(cpath.read_text()).items()}
    ds = MXUGraphDataset(f"{DATA}/{data_dir}", mxu_arm=arm)
    models = load_models(paths, arm, dev)
    groups = defaultdict(list)
    for flat, (fi, _) in enumerate(ds._index):
        groups[fi].append(flat)
    out = {}
    for fi, flats in groups.items():
        name = Path(ds._file_paths[fi]).stem
        if name not in names:
            continue
        rt = np.asarray(ds._get_handle(fi)["config_runtime"], dtype=np.float64)[[ds._index[i][1] for i in flats]]
        preds = {s: [] for s in models}
        for start in range(0, len(flats), bs):
            batch = Batch.from_data_list([ds[i] for i in flats[start:start + bs]]).to(dev)
            for s, m in models.items():
                r = _forward(m, batch)
                preds[s].append((r[0] if isinstance(r, tuple) else r).cpu())
        out[name] = {s: float(rt[int(np.argmax(torch.cat(preds[s]).numpy()))] / rt.min()) for s in models}
        print(f"  {tag:<22} {data_dir:<14} b{bs:<3} {name[:40]:<42} "
              + "  ".join(f"{out[name][s]:.3f}" for s in SEEDS), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(out))
    return out


def read_cache(p):
    return {g: {int(s): v for s, v in d.items()} for g, d in json.loads(p.read_text()).items()}


def per_seed_mean(raw, names=None):
    """raw = {"coll::graph": {seed: x}} -> per-seed mean over instances whose graph is in `names`."""
    keys = [k for k in raw if names is None or k.split("::")[1] in names]
    return [float(np.mean([raw[k][s] for k in keys])) for s in SEEDS]


def stats(a, b):
    ps = paired_stats(a, b)
    d = np.array(a) - np.array(b)
    sd = float(d.std(ddof=1))
    ps["mde"] = min_detectable_effect(sd, len(SEEDS)) if sd > 0 else float("nan")
    ps["per_seed_delta"] = d.round(4).tolist()
    return ps


def row(label, r):
    return (f"| {label} | {r['delta']:+.4f} | {r['p_t']:.3g} | {r['p_w']:.3g} | "
            f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['d']:.2f} | {r['mde']:.4f} |")


HDR = ["| contrast (negative = first arm closer to optimal) | Δ | t-p | Wilcoxon-p | 95% CI | d | MDE |",
       "|---|---|---|---|---|---|---|"]
fmt = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"


def run_r3(dev):
    paths = {s: Path(f"checkpoints/r3/xla_random/mxu-scramble-gpu-seed{s}/best_model.pt") for s in SEEDS}
    names = CONV_GRAPHS | OTHER_GRAPHS
    raw = {32: {}, 8: {}}
    for bs in (32, 8):
        src = REG_8C if bs == 32 else O19_B8
        raw[bs]["scramble-gpu"] = {}
        for coll in COLLECTIONS:
            for g, sv in score("r3_scramble-gpu", paths, "scramble", f"{coll}/valid", names, bs, dev).items():
                raw[bs]["scramble-gpu"][f"{coll}::{g}"] = sv
        for arm in ("baseline", "channel", "scramble"):
            raw[bs][arm] = {f"{coll}::{g}": sv for coll in COLLECTIONS
                            for g, sv in read_cache(src / f"{arm}_{coll.replace('/', '_')}.json").items()}
    m = {bs: {a: {"conv": per_seed_mean(raw[bs][a], CONV_GRAPHS), "other": per_seed_mean(raw[bs][a], OTHER_GRAPHS)}
              for a in raw[bs]} for bs in raw}
    tests = [("R3-P4: channel − scramble-GPU", "channel", "scramble-gpu", "conv"),
             ("scramble-GPU − baseline", "scramble-gpu", "baseline", "conv"),
             ("placebo: scramble-GPU − baseline", "scramble-gpu", "baseline", "other"),
             ("hardware offset: scramble-GPU − scramble-CPU", "scramble-gpu", "scramble", "conv"),
             ("hardware offset: scramble-GPU − scramble-CPU", "scramble-gpu", "scramble", "other"),
             ("reference (registered P4): channel − scramble-CPU", "channel", "scramble", "conv")]
    res = {"means": {str(b): m[b] for b in m}, "contrasts": {}, "raw": {str(b): raw[b] for b in raw}}
    L = ["# R3 — P4 at matched hardware: scramble arm re-trained on GPU (replication)", "",
         "Frozen: `r3_scramble_gpu_replication_declaration.md`. lr 1e-3, 5 seeds, 8c frozen split "
         "(CONV 3 graphs × 2 xla collections, OTHER 4 × 2). **Primary = eval batch 32 (as P4)**; batch 8 = "
         "sensitivity. 8c arms read from the registered caches (b32) and O19's caches (b8). The registered "
         "P4 (scramble on CPU) stays primary; R3 is reported beside it.", ""]
    for bs in (32, 8):
        L += [f"## Eval batch {bs}{' (primary)' if bs == 32 else ' (sensitivity)'}", "",
              "| arm | CONV Top-1 | OTHER Top-1 |", "|---|---|---|"]
        for a in ("baseline", "channel", "scramble", "scramble-gpu"):
            lab = {"scramble": "scramble (CPU, 8c)", "scramble-gpu": "scramble (GPU, R3)"}.get(a, f"{a} (GPU, 8c)")
            L.append(f"| {lab} | {fmt(m[bs][a]['conv'])} | {fmt(m[bs][a]['other'])} |")
        L += [""] + HDR
        for key, a1, a2, sub in tests:
            r = stats(m[bs][a1][sub], m[bs][a2][sub])
            res["contrasts"][f"b{bs}:{key}:{sub}"] = r
            L.append(row(f"{key}, {sub.upper()}", r))
        L.append("")
    p4 = res["contrasts"]["b32:R3-P4: channel − scramble-GPU:conv"]
    verdict = ("P4 replicates at matched hardware" if p4["delta"] < 0 and p4["p_t"] < .05 else
               "P4 holds as registered but did not replicate significantly at matched hardware" if p4["delta"] < 0
               else "P4 does not survive matched hardware")
    plac = {bs: res["contrasts"][f"b{bs}:placebo: scramble-GPU − baseline:other"] for bs in (32, 8)}
    L += ["## Verdict (declaration §3 table, batch 32)", "",
          f"- R3-P4 Δ {p4['delta']:+.4f}, p = {p4['p_t']:.3g}, MDE {p4['mde']:.4f} → **{verdict}**.",
          f"- OTHER placebo on GPU: Δ {plac[32]['delta']:+.4f}, p = {plac[32]['p_t']:.3g} (b8: {plac[8]['delta']:+.4f}, "
          f"p = {plac[8]['p_t']:.3g}). The registered CPU-arm placebo was +0.0079, p = .028. "
          + ("The placebo also moves on GPU, so the CPU-offset explanation is dropped (declaration §3)."
             if plac[32]["p_t"] < .05 else "It does not move on GPU, consistent with the CPU-offset explanation.")]
    res["verdict"] = verdict
    return "r3", L, res


def run_r4(dev):
    import subprocess
    for arm in ("baseline", "channel"):
        star = subprocess.run([sys.executable, "scripts/phase8e_lr_star.py", "--arm", arm, "--print-lr-star"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert star == R4_LR, (arm, star)
    raw = {32: {}, 8: {}}
    for bs in (32, 8):
        for arm in ("baseline", "channel"):
            paths = {s: Path(f"checkpoints/r4/xla_random_r4/mxu-{arm}-lr{R4_LR}-seed{s}/best_model.pt") for s in SEEDS}
            raw[bs][arm] = {"held": {}, "other": {}}
            for coll in COLLECTIONS:
                for g, sv in score(f"r4_{arm}", paths, arm, f"{coll}/train", R4_HELD_OUT, bs, dev).items():
                    raw[bs][arm]["held"][f"{coll}::{g}"] = sv
                for g, sv in score(f"r4_{arm}", paths, arm, f"{coll}/valid", OTHER_GRAPHS, bs, dev).items():
                    raw[bs][arm]["other"][f"{coll}::{g}"] = sv
    n_inst = len(raw[32]["baseline"]["held"])
    assert n_inst == 19, n_inst
    res = {"lr_star": R4_LR, "contrasts": {}, "raw": {str(b): raw[b] for b in raw}}
    L = ["# R4 — Channel effect on held-out convolution families (exploratory)", "",
         f"Frozen: `r4_heldout_conv_declaration.md` + Amendment 1. Both arms at their 8e LR* = {R4_LR}, trained on "
         "58 xla:random train graphs (11 conv graphs from 6 families held out), 5 seeds. Evaluation: 19 held-out "
         "instances (11 xla:random + 8 xla:default train-split files). **Primary = eval batch 32**; batch 8 = "
         "sensitivity. Read beside 8e Q1 (the tuned contrast), not 8c's lr-1e-3 contrast.", ""]
    for bs in (32, 8):
        mb = {a: per_seed_mean(raw[bs][a]["held"]) for a in ("baseline", "channel")}
        mo = {a: per_seed_mean(raw[bs][a]["other"]) for a in ("baseline", "channel")}
        prim, plac = stats(mb["channel"], mb["baseline"]), stats(mo["channel"], mo["baseline"])
        res["contrasts"][f"b{bs}:primary"], res["contrasts"][f"b{bs}:placebo"] = prim, plac
        L += [f"## Eval batch {bs}{' (primary)' if bs == 32 else ' (sensitivity)'}", "",
              "| arm | held-out CONV Top-1 (19 instances) | OTHER Top-1 (8c placebo) |", "|---|---|---|",
              f"| baseline | {fmt(mb['baseline'])} | {fmt(mo['baseline'])} |",
              f"| channel | {fmt(mb['channel'])} | {fmt(mo['channel'])} |", ""] + HDR + [
              row("primary: channel − baseline, held-out CONV", prim),
              row("placebo: channel − baseline, OTHER", plac), "",
              "Per family (mean over its instances, then over seeds) and leave-one-family-out:", "",
              "| family | instances | baseline | channel | Δ | drop this family: Δ, t-p |", "|---|---|---|---|---|---|"]
        for fam, gs in R4_FAMILIES.items():
            fb = per_seed_mean(raw[bs]["baseline"]["held"], set(gs))
            fc = per_seed_mean(raw[bs]["channel"]["held"], set(gs))
            rest = R4_HELD_OUT - set(gs)
            lo = stats(per_seed_mean(raw[bs]["channel"]["held"], rest), per_seed_mean(raw[bs]["baseline"]["held"], rest))
            ni = sum(k.split("::")[1] in gs for k in raw[bs]["baseline"]["held"])
            res["contrasts"][f"b{bs}:family:{fam}"] = {"baseline": fb, "channel": fc, "leave_out": lo}
            L.append(f"| {fam} | {ni} | {np.mean(fb):.4f} | {np.mean(fc):.4f} | {np.mean(fc) - np.mean(fb):+.4f} | "
                     f"{lo['delta']:+.4f}, {lo['p_t']:.3g} |")
        if bs == 32:
            L += ["", "Per instance (mean over seeds):", "", "| instance | baseline | channel | Δ |", "|---|---|---|---|"]
            for k in sorted(raw[bs]["baseline"]["held"]):
                b_, c_ = (np.mean(list(raw[bs][a]["held"][k].values())) for a in ("baseline", "channel"))
                L.append(f"| {k.replace('::', ' · ')} | {b_:.4f} | {c_:.4f} | {c_ - b_:+.4f} |")
        L.append("")
    p = res["contrasts"]["b32:primary"]
    verdict = ("the effect generalises (exploratory, six held-out families)" if p["delta"] < 0 and p["p_t"] < .05 else
               "direction consistent on held-out families, not significant at n = 5" if p["delta"] < 0 else
               "the effect does not generalise beyond the Inception/ResNet families tested in 8c")
    res["verdict"] = verdict
    L += ["## Verdict (declaration §4 table, batch 32)", "",
          f"- Δ {p['delta']:+.4f}, p = {p['p_t']:.3g}, MDE {p['mde']:.4f} → **{verdict}**."]
    return "r4", L, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["r3", "r4"], required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    name, L, res = (run_r3 if a.which == "r3" else run_r4)(dev)
    out = Path(f"experiments/{name}_results.md")
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L), f"\nWrote {out} and {out.with_suffix('.json')}", sep="\n")


if __name__ == "__main__":
    main()
