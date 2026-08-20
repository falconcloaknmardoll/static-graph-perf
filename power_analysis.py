#!/usr/bin/env python3
"""
Power-analysis tool for paired-seed Kendall's-tau comparisons (paper_outline.md §4, the
"rigor backbone" release artifact).

Formalizes two things this project established empirically (VERIFICATION.md C9, and the
"tightening curve" referenced in paper_outline.md §5):

  1. Minimum detectable effect (MDE): given the observed seed-to-seed noise (sigma) on a
     paired architectural comparison, how large does a true tau gain need to be before a
     study with n_seeds seeds can reliably detect it (paired t-test, two-sided)?

  2. The tightening curve: given real per-graph tau measurements across seeds, how does the
     seed-to-seed standard deviation of the AGGREGATE (mean-over-graphs) tau shrink as the
     number of evaluation graphs grows? Answers "how many eval graphs do you need before a
     5-seed comparison has any power at all" -- the direct rebuttal to "you only have 7-20
     validation graphs" (paper_outline.md's own anticipated reviewer objection).

Usage
-----
  # MDE table for the paper's own established noise floor (C9: sigma in [0.04, 0.07])
  python power_analysis.py mde --sigma 0.04 0.055 0.07 --n-seeds 3 5 7 10

  # Empirical tightening curve from real per-graph tau data (e.g. phase6b_confound_checks.json)
  python power_analysis.py tighten --per-graph-json experiments/phase6b_confound_checks.json \\
      --key per_graph_tau.wide-baseline --out experiments/power_analysis_results.md

Both subcommands also accept --figure to write a PNG (matplotlib, optional dependency).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sp_stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. Minimum detectable effect (paired t-test, two-sided)
# ─────────────────────────────────────────────────────────────────────────────

def min_detectable_effect(sigma: float, n_seeds: int, alpha: float = 0.05,
                           power: float = 0.8) -> float:
    """MDE for a paired-difference test with n_seeds paired observations (seeds), given the
    observed paired-difference standard deviation `sigma`.

    Standard two-sided paired-t sample-size relation, solved for effect size:
        MDE = (t_{1-alpha/2, df} + t_{1-power, df}) * sigma / sqrt(n_seeds)
    where df = n_seeds - 1. This is the exact-t (not normal-approximation) version, appropriate
    for the small seed counts (n=5) used throughout this project.
    """
    if n_seeds < 2:
        return float("inf")
    df = n_seeds - 1
    t_alpha = sp_stats.t.ppf(1 - alpha / 2, df)
    t_power = sp_stats.t.ppf(power, df)
    return (t_alpha + t_power) * sigma / np.sqrt(n_seeds)


def required_seeds(effect: float, sigma: float, alpha: float = 0.05, power: float = 0.8,
                    max_seeds: int = 200) -> int:
    """Smallest n_seeds such that min_detectable_effect(sigma, n_seeds) <= effect."""
    for n in range(2, max_seeds + 1):
        if min_detectable_effect(sigma, n, alpha, power) <= effect:
            return n
    return -1  # not achievable within max_seeds


def mde_table(sigmas: Sequence[float], n_seeds_list: Sequence[int],
              alpha: float = 0.05, power: float = 0.8) -> List[Dict]:
    rows = []
    for sigma in sigmas:
        for n in n_seeds_list:
            rows.append({
                "sigma": sigma, "n_seeds": n,
                "mde": min_detectable_effect(sigma, n, alpha, power),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tightening curve: aggregate-tau seed-std vs #eval-graphs (bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def tightening_curve(per_graph_tau: Dict[str, List[float]], graph_counts: Sequence[int],
                      n_bootstrap: int = 2000, seed: int = 0) -> List[Dict]:
    """per_graph_tau: {graph_name: [tau_seed0, tau_seed1, ...]} (same seed order for all
    graphs, i.e. a rectangular graphs x seeds matrix). For each candidate #graphs G in
    graph_counts, bootstrap-resample G graphs (with replacement, from the available pool),
    compute the aggregate (mean-over-graphs) tau per seed, then the across-seed std of that
    aggregate. Repeat n_bootstrap times; report the median and 90% interval of that std.

    This directly answers: "if we had evaluated on G graphs instead of however many we
    actually have, how noisy would the reported mean-tau be, seed to seed?"
    """
    names = list(per_graph_tau.keys())
    n_seeds = len(next(iter(per_graph_tau.values())))
    mat = np.array([per_graph_tau[n] for n in names])  # (n_graphs_available, n_seeds)
    n_available = mat.shape[0]
    rng = np.random.default_rng(seed)

    out = []
    for G in graph_counts:
        stds = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_available, size=G)
            agg_per_seed = mat[idx].mean(axis=0)  # (n_seeds,)
            stds.append(float(np.std(agg_per_seed, ddof=1)) if n_seeds > 1 else 0.0)
        stds = np.array(stds)
        out.append({
            "n_graphs": G,
            "std_median": float(np.median(stds)),
            "std_p05": float(np.percentile(stds, 5)),
            "std_p95": float(np.percentile(stds, 95)),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_mde(args):
    rows = mde_table(args.sigma, args.n_seeds, args.alpha, args.power)
    print(f"Minimum detectable effect (paired t-test, alpha={args.alpha}, power={args.power})")
    print(f"{'sigma':>8} {'n_seeds':>8} {'MDE (tau)':>10}")
    for r in rows:
        print(f"{r['sigma']:>8.3f} {r['n_seeds']:>8d} {r['mde']:>10.4f}")

    print(f"\nRequired seeds to detect specific effect sizes (at worst-case sigma={max(args.sigma):.3f}):")
    for eff in [0.02, 0.05, 0.10, 0.15]:
        n = required_seeds(eff, max(args.sigma), args.alpha, args.power)
        print(f"  effect=Δτ={eff:.2f}  ->  n_seeds={n}")

    if args.out:
        lines = ["# Power analysis — minimum detectable effect\n",
                 f"Paired t-test, alpha={args.alpha}, power={args.power}.\n",
                 f"\n| sigma | n_seeds | MDE (Δτ) |\n|---|---|---|\n"]
        for r in rows:
            lines.append(f"| {r['sigma']:.3f} | {r['n_seeds']} | {r['mde']:.4f} |\n")
        Path(args.out).write_text("".join(lines))
        print(f"\nWrote {args.out}")


def _cmd_tighten(args):
    data = json.loads(Path(args.per_graph_json).read_text())
    node = data
    for k in args.key.split("."):
        node = node[k]
    per_graph_tau = node  # {graph_name: [tau_per_seed...]}

    max_available = len(per_graph_tau)
    graph_counts = list(range(2, args.max_graphs + 1)) if args.max_graphs else list(range(2, 21))
    rows = tightening_curve(per_graph_tau, graph_counts, n_bootstrap=args.n_bootstrap)

    print(f"Tightening curve (bootstrap from {max_available} available graphs, "
          f"{args.n_bootstrap} resamples/point):")
    print(f"{'n_graphs':>9} {'std_median':>11} {'std_p05':>9} {'std_p95':>9}")
    for r in rows:
        print(f"{r['n_graphs']:>9d} {r['std_median']:>11.4f} {r['std_p05']:>9.4f} {r['std_p95']:>9.4f}")

    if args.out:
        lines = ["# Power analysis — tightening curve\n",
                 f"Aggregate-tau seed-std vs #eval-graphs, bootstrap-resampled from "
                 f"{max_available} real per-graph tau measurements ({args.n_bootstrap} draws/point).\n",
                 "\n| n_graphs | std (median) | std (p05) | std (p95) |\n|---|---|---|---|\n"]
        for r in rows:
            lines.append(f"| {r['n_graphs']} | {r['std_median']:.4f} | {r['std_p05']:.4f} | {r['std_p95']:.4f} |\n")
        Path(args.out).write_text("".join(lines))
        print(f"\nWrote {args.out}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [r["n_graphs"] for r in rows]
        med = [r["std_median"] for r in rows]
        p05 = [r["std_p05"] for r in rows]
        p95 = [r["std_p95"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, med, color="#2563eb", lw=2, label="median")
        ax.fill_between(ns, p05, p95, color="#2563eb", alpha=0.15, label="p05-p95")
        ax.set_xlabel("# evaluation graphs")
        ax.set_ylabel("seed-to-seed std of aggregate tau")
        ax.set_title("Tightening curve: noise floor vs eval-set size")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"Wrote {args.figure}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_mde = sub.add_parser("mde", help="Minimum detectable effect table")
    p_mde.add_argument("--sigma", type=float, nargs="+", default=[0.04, 0.055, 0.07],
                        help="Observed paired-difference seed std (default: C9's 0.04-0.07 range)")
    p_mde.add_argument("--n-seeds", type=int, nargs="+", default=[3, 5, 7, 10, 15, 20])
    p_mde.add_argument("--alpha", type=float, default=0.05)
    p_mde.add_argument("--power", type=float, default=0.8)
    p_mde.add_argument("--out", type=str, default=None)
    p_mde.set_defaults(func=_cmd_mde)

    p_t = sub.add_parser("tighten", help="Empirical tightening curve from real per-graph tau data")
    p_t.add_argument("--per-graph-json", type=str, required=True,
                      help="JSON file containing {graph: [tau_per_seed...]} at --key")
    p_t.add_argument("--key", type=str, required=True,
                      help="Dotted path into the JSON to the per-graph-tau dict")
    p_t.add_argument("--max-graphs", type=int, default=20)
    p_t.add_argument("--n-bootstrap", type=int, default=2000)
    p_t.add_argument("--out", type=str, default=None)
    p_t.add_argument("--figure", type=str, default=None)
    p_t.set_defaults(func=_cmd_tighten)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
