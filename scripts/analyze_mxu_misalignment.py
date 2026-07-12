#!/usr/bin/env python3
"""
Phase 8 evidence — does MXU misalignment explain the convolution-failure (C10)?

Connects two so-far-separate findings WITHOUT the model and WITHOUT the 8c retrain:
  • Phase 5: convolution graphs fail (~47% off Top-1) vs transformer/UNet (~5%).
  • Phase 8: MXU-alignment features (channel/contraction vs 128×128 systolic array).

We compute, per graph, a compute-weighted MXU padding-overhead score from the
static conv shapes, and — crucially — the layout SENSITIVITY measured directly
from the real `config_runtime` array (model-independent). Three tests:

  T1 (confirmatory): MXU overhead vs Phase-5 Top-1 slowdown across the 7 val
     graphs. Expected strong, but partly tautological (only conv graphs have a
     defined overhead) — reported as such.
  T2 (within-group, n=3): among the 3 conv graphs, does overhead ORDER match the
     slowdown order (resnet50 worst)? A real, if tiny, mechanism test.
  T3 (the non-circular one): does layout sensitivity (runtime spread, from real
     runtimes) track conv-ness / MXU overhead across ALL xla graphs (n≫7)? This
     links the static MXU feature to genuine layout sensitivity independent of
     the GNN.

Outputs: experiments/phase8_mxu_misalignment.md (+ .json).
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxu_features import compute_mxu_features, CONV_OPCODE, MXU_DIM  # noqa: E402

# Phase 5 ground truth (experiments/phase5_error_analysis.md): file → (family, slowdown)
PHASE5 = {
    "unet_3d.4x4.bf16":                    ("memory-bound", 1.010),
    "tf2_bert_pretrain_dynamic_batch_size":("transformer",  1.056),
    "bert_pretraining.4x4.fp16":           ("transformer",  1.067),
    "mlperf_bert_batch_24_2x2":            ("transformer",  1.072),
    "inception_v3_batch_128_train":        ("convolution",  1.441),
    "resnet_v1_50_official_batch_128_bf16":("convolution",  1.452),
    "resnet50.4x4.fp16":                   ("convolution",  1.512),
}
LAYOUT_ROOT = "data/predict-ai-model-runtime/npz_all/npz/layout"


def runtime_sensitivity(z) -> dict:
    """Layout sensitivity from REAL runtimes (no model). Higher ⇒ layout matters more."""
    r = np.asarray(z["config_runtime"], dtype=np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size < 8:
        return {"rt_cv": np.nan, "rt_p90_p10": np.nan, "n_cfg": int(r.size)}
    p10, p90 = np.percentile(r, [10, 90])
    return {"rt_cv": float(r.std() / r.mean()),
            "rt_p90_p10": float(p90 / max(p10, 1e-9)),
            "n_cfg": int(r.size)}


def mxu_profile(path: str) -> dict:
    """Per-graph MXU padding-overhead, compute-weighted over conv nodes."""
    z = np.load(path, mmap_mode="r")
    op = np.asarray(z["node_opcode"])
    feats, names = compute_mxu_features(np.asarray(z["node_feat"]), op,
                                        np.asarray(z["edge_index"]))
    ci = {n: i for i, n in enumerate(names)}
    conv = np.where(op == CONV_OPCODE)[0]
    n_nodes = int(op.shape[0])
    out = {"n_nodes": n_nodes, "n_conv": int(conv.size),
           "mxu_frac": conv.size / max(n_nodes, 1)}
    out.update(runtime_sensitivity(z))

    if conv.size == 0:
        out.update({"conv_mac_frac": 0.0, "mxu_overhead": np.nan,
                    "cout_util": np.nan, "k_util": np.nan, "resolved": 0})
        return out

    cu = feats[conv, ci["c_out_mxu_util"]].astype(np.float64)
    ku = feats[conv, ci["k_mxu_util"]].astype(np.float64)
    macs = np.power(10.0, feats[conv, ci["log10_macs"]].astype(np.float64))
    resolved = ~np.isnan(ku)
    # overhead_i = padded_cycles/useful = 1/(cout_util·k_util); compute-weighted.
    eff = cu * np.where(resolved, ku, 1.0)            # k_util=1 fallback if unresolved
    eff = np.clip(eff, 1e-6, 1.0)
    overhead = float(np.sum(macs / eff) / np.sum(macs))   # ≥1; 1=perfect packing
    out.update({
        "conv_mac_frac": float(np.sum(macs) / max(np.sum(macs), 1e-9)),  # within-graph 1.0 (conv only here)
        "mxu_overhead": overhead,
        "cout_util": float(np.average(cu, weights=macs)),
        "k_util": float(np.average(ku[resolved], weights=macs[resolved])) if resolved.any() else np.nan,
        "resolved": int(resolved.sum()),
    })
    return out


def spearman(x, y) -> tuple:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return (np.nan, x.size)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return (np.nan, x.size)
    return (float(np.corrcoef(rx, ry)[0, 1]), x.size)


def main() -> int:
    lines = []
    def P(s=""): print(s); lines.append(s)

    # ── The 7 Phase-5 val graphs ──────────────────────────────────────────────
    P("# Phase 8 — MXU misalignment vs the convolution failure\n")
    P("## Test set 1: the 7 Phase-5 validation graphs\n")
    P(f"{'graph':<26}{'family':<13}{'slow':>6}{'mxu_frac':>9}"
      f"{'overhead':>9}{'cout_u':>7}{'k_u':>6}{'rt_cv':>7}{'rt_p90/10':>10}")
    rows = []
    for key, (fam, slow) in sorted(PHASE5.items(), key=lambda kv: kv[1][1]):
        f = glob.glob(f"{LAYOUT_ROOT}/**/{key}.npz", recursive=True)
        if not f:
            P(f"{key:<26}  NOT FOUND"); continue
        p = mxu_profile(f[0]); p["graph"] = key; p["family"] = fam; p["slow"] = slow
        rows.append(p)
        ov = "  n/a" if np.isnan(p["mxu_overhead"]) else f"{p['mxu_overhead']:.3f}"
        ku = "  n/a" if np.isnan(p["k_util"]) else f"{p['k_util']:.2f}"
        cu = "  n/a" if np.isnan(p["cout_util"]) else f"{p['cout_util']:.2f}"
        P(f"{key[:25]:<26}{fam:<13}{slow:>6.3f}{p['mxu_frac']:>9.4f}"
          f"{ov:>9}{cu:>7}{ku:>6}{p['rt_cv']:>7.2f}{p['rt_p90_p10']:>10.2f}")

    conv = [r for r in rows if r["family"] == "convolution"]
    other = [r for r in rows if r["family"] != "convolution"]

    # ── T1 confirmatory ───────────────────────────────────────────────────────
    P("\n## T1 (confirmatory): MXU signal vs Phase-5 slowdown across 7 graphs")
    rho, n = spearman([r["mxu_frac"] for r in rows], [r["slow"] for r in rows])
    P(f"  mxu_frac     vs slowdown: Spearman ρ={rho:+.2f}  (n={n})")
    rho_cv, _ = spearman([r["rt_cv"] for r in rows], [r["slow"] for r in rows])
    P(f"  runtime CV   vs slowdown: Spearman ρ={rho_cv:+.2f}  (n={n})  "
      "← real-runtime layout sensitivity, model-independent")
    P("  (mxu_frac/overhead are only defined where conv exists, so T1 partly")
    P("   restates the conv-vs-not split; the load-bearing tests are T2/T3.)")

    # ── T2 within conv ────────────────────────────────────────────────────────
    P("\n## T2 (mechanism, within the 3 convolution graphs)")
    if len(conv) == 3:
        cs = sorted(conv, key=lambda r: r["slow"])
        for r in cs:
            P(f"  {r['graph'][:34]:<35} slowdown={r['slow']:.3f}  "
              f"MXU overhead={r['mxu_overhead']:.3f}  cout_util={r['cout_util']:.2f}  "
              f"k_util={r['k_util']:.2f}")
        order_match = ([r["graph"] for r in sorted(conv, key=lambda r: r["slow"])]
                       == [r["graph"] for r in sorted(conv, key=lambda r: r["mxu_overhead"])])
        P(f"  → does higher MXU overhead track higher slowdown? "
          f"{'YES (orders match, n=3)' if order_match else 'NO (orders differ)'}")

    # ── T3 dataset-wide, model-independent ────────────────────────────────────
    P("\n## T3 (the non-circular test): layout sensitivity vs MXU across ALL xla graphs")
    allp = []
    for f in glob.glob(f"{LAYOUT_ROOT}/xla/*/*/*.npz"):
        try:
            allp.append(mxu_profile(f))
        except Exception:
            pass
    P(f"  scanned {len(allp)} xla graphs (train+valid, default+random).")
    have_conv = [r for r in allp if r["n_conv"] > 0]
    no_conv = [r for r in allp if r["n_conv"] == 0]
    def med(rs, k):
        v = [r[k] for r in rs if np.isfinite(r[k])]; return float(np.median(v)) if v else np.nan
    P(f"  layout sensitivity (runtime CV)  — median:")
    P(f"     conv-bearing graphs (n={len(have_conv)}): {med(have_conv,'rt_cv'):.3f}")
    P(f"     conv-free graphs    (n={len(no_conv)}): {med(no_conv,'rt_cv'):.3f}")
    rho_a, na = spearman([r["mxu_frac"] for r in allp], [r["rt_cv"] for r in allp])
    P(f"  mxu_frac vs runtime CV across all graphs: Spearman ρ={rho_a:+.2f} (n={na})")
    rho_o, no_ = spearman([r["mxu_overhead"] for r in have_conv], [r["rt_cv"] for r in have_conv])
    P(f"  MXU overhead vs runtime CV (conv graphs only): Spearman ρ={rho_o:+.2f} (n={no_})")

    P("\n## Honest caveats")
    P("  • T1/T2 are n=7 / n=3 — directional, not significance. Same post-hoc")
    P("    grouping caveat as Phase 5 (C10 is exploratory).")
    P("  • MXU overhead is a STATIC property of conv shapes; it explains why conv")
    P("    graphs are a distinct regime, but whether ENCODING it helps the model")
    P("    is the Phase-8c retrain question — not settled here.")
    P("  • T3 uses real runtimes (no model), so it is the most defensible: it asks")
    P("    whether convolution-heavy graphs are genuinely more layout-sensitive.")

    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/phase8_mxu_misalignment.md").write_text("\n".join(lines) + "\n")
    Path("experiments/phase8_mxu_misalignment.json").write_text(json.dumps(
        {"phase5_graphs": rows, "n_xla_scanned": len(allp)}, indent=2, default=float))
    P("\nWrote experiments/phase8_mxu_misalignment.{md,json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
