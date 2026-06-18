#!/usr/bin/env python3
"""
batch_topology_correlation.py
------------------------------
Phase 0.5 – Statistical Correlation Analysis.

Profiles every unique XLA computation graph across the layout:xla:random
train/valid/test splits (84 graphs total) and correlates graph topology
metrics with compiler-runtime variance.

Metrics computed per graph
──────────────────────────
  N              – number of nodes
  lambda_1       – spectral gap (algebraic connectivity of the Normalized
                   Laplacian); small λ₁ ↔ structural bottleneck
  diameter       – approximate graph diameter (lower bound, double-sweep BFS)
  avg_clustering – average clustering coefficient C
  runtime_mean   – mean config_runtime across all compiler configurations
  runtime_std    – standard deviation of config_runtime
  runtime_cv     – coefficient of variation (std/mean); scale-free variance
                   proxy that is comparable across workloads
  n_configs      – number of compiler configurations for this graph

Outputs
───────
  scripts/topology_metrics.csv          – raw metrics DataFrame
  scripts/fig1_spectral_gap_vs_cv.png   – λ₁ vs Runtime CV (log x, trendline)
  scripts/fig2_diameter_vs_cv.png       – Diameter vs Runtime CV  (trendline)
  scripts/fig3_N_vs_spectral_gap.png    – N vs λ₁ (shows structural scaling)
  scripts/fig4_correlation_heatmap.png  – Spearman ρ heatmap over all metrics

Usage:
    python batch_topology_correlation.py
    python batch_topology_correlation.py --diameter-sweeps 5
    python batch_topology_correlation.py --splits train      # train only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")                   # non-interactive backend; must be before pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.sparse.csgraph import (
    connected_components as sp_connected_components,
    dijkstra,
    laplacian as sp_laplacian,
)
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.stats import spearmanr

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
os.makedirs(SCRIPT_DIR, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
_SPLIT_DIRS = {
    "train": os.path.join(
        "data", "predict-ai-model-runtime",
        "npz_all", "npz", "layout", "xla", "random", "train",
    ),
    "valid": os.path.join(
        "data", "predict-ai-model-runtime",
        "npz_all", "npz", "layout", "xla", "random", "valid",
    ),
    "test": os.path.join(
        "data", "predict-ai-model-runtime",
        "npz_all", "npz", "layout", "xla", "random", "test",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphProfile:
    name:            str
    split:           str
    N:               int
    E_directed:      int
    lcc_fraction:    float
    lambda_1:        float          # NaN if eigsh failed
    diameter:        int
    avg_clustering:  float
    runtime_mean:    float
    runtime_std:     float
    runtime_cv:      float          # std / mean; NaN if mean == 0
    n_configs:       int
    elapsed_s:       float


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_symmetric_csr(src: np.ndarray, dst: np.ndarray, N: int) -> sp.csr_matrix:
    """
    Build a binary symmetric scipy CSR adjacency matrix.

    Symmetrizes src→dst and dst→src; removes self-loops; clamps weights to 1.
    """
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.data = np.ones_like(A.data)
    A.eliminate_zeros()
    return A


def _lcc_of_csr(A: sp.csr_matrix) -> tuple[sp.csr_matrix, float]:
    """
    Extract the Largest Connected Component of a symmetric CSR matrix.

    Returns (A_lcc, lcc_fraction) where A_lcc is re-indexed to [0, |LCC|-1].
    """
    n_comp, labels = sp_connected_components(A, directed=False, connection="weak")
    if n_comp == 1:
        return A, 1.0

    comp_sizes = np.bincount(labels)
    lcc_label  = int(np.argmax(comp_sizes))
    lcc_mask   = labels == lcc_label
    lcc_frac   = float(lcc_mask.sum()) / A.shape[0]

    # Slice rows and columns; re-indexing is implicit since csr[mask][:,mask]
    # produces a compact matrix.
    A_lcc = A[lcc_mask][:, lcc_mask]
    A_lcc = A_lcc.tocsr()
    A_lcc.data = np.ones_like(A_lcc.data)
    return A_lcc, lcc_frac


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 – Spectral gap (scipy eigsh, Normalized Laplacian)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_gap(A: sp.csr_matrix, k: int = 6) -> float:
    """
    Return λ₁ (second-smallest eigenvalue of the Normalized Laplacian).

    Uses ARPACK's Lanczos solver via eigsh; O(k × nnz × iterations).

    Numeric stability:
      • Request k eigenvalues; take absolute values to absorb floating-point
        negatives near zero; sort ascending; λ₀ ≈ 0 is first, λ₁ is second.

    Raises ArpackNoConvergence if the solver fails to converge — caller
    should catch this and record NaN rather than crashing.
    """
    n = A.shape[0]
    k = min(k, n - 2)
    if k < 2:
        return float("nan")

    L = sp_laplacian(A, normed=True)
    vals, _ = eigsh(L, k=k, which="SM", tol=1e-5, maxiter=10_000)
    vals = np.sort(np.abs(vals.real))
    return float(vals[1])


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 – Approximate diameter (scipy Dijkstra double-sweep)
# ─────────────────────────────────────────────────────────────────────────────

def _scipy_eccentricity(A: sp.csr_matrix, source: int) -> tuple[int, int]:
    """
    BFS from `source` via scipy Dijkstra (unweighted=True → pure BFS, O(N+E)).

    Returns (eccentricity, farthest_node_index).
    """
    dists = dijkstra(A, directed=False, indices=source, unweighted=True)
    finite = np.isfinite(dists)
    if not finite.any():
        return 0, source
    farthest = int(np.argmax(dists))
    return int(dists[farthest]), farthest


def compute_approx_diameter(
    A: sp.csr_matrix, n_sweeps: int = 3, seed: int = 42
) -> int:
    """
    Lower bound on the graph diameter via iterated double-sweep BFS.

    Per sweep:
      1.  Start from a random node u.
      2.  BFS(u) → farthest node v   (first peripheral node)
      3.  BFS(v) → eccentricity ecc  (tight lower bound on diameter)
    Return the maximum ecc over all sweeps.

    scipy Dijkstra with unweighted=True runs as O(N+E) BFS in C; even
    for N=43 K this takes < 5 ms per sweep.
    """
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    best = 0

    for _ in range(n_sweeps):
        u = int(rng.integers(0, N))
        _, v      = _scipy_eccentricity(A, u)
        ecc, _    = _scipy_eccentricity(A, v)
        best = max(best, ecc)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 – Average clustering coefficient (NetworkX, O(N × avg_deg²))
# ─────────────────────────────────────────────────────────────────────────────

def compute_avg_clustering(src: np.ndarray, dst: np.ndarray, N: int) -> float:
    """Build an undirected NetworkX graph from raw (src, dst) arrays and
    compute the average clustering coefficient."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    return nx.average_clustering(G)


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_one_graph(
    npz_path: str,
    split: str,
    diameter_sweeps: int,
) -> Optional[GraphProfile]:
    """
    Load one .npz file and compute all topology and runtime metrics.

    Returns None if the file is malformed or any non-recoverable error occurs.
    λ₁ is set to NaN if eigsh fails to converge; the graph is still included.
    """
    t0 = time.perf_counter()
    name = os.path.basename(npz_path)

    try:
        npz = np.load(npz_path, mmap_mode="r")

        # ── Graph structure ──────────────────────────────────────────────
        ei  = np.array(npz["edge_index"], dtype=np.int64)  # (E, 2)
        src = ei[:, 0]
        dst = ei[:, 1]
        N   = int(npz["node_feat"].shape[0])
        E   = len(src)

        A          = _build_symmetric_csr(src, dst, N)
        A_lcc, lcc_frac = _lcc_of_csr(A)

        # ── Spectral gap ─────────────────────────────────────────────────
        try:
            lambda_1 = compute_spectral_gap(A_lcc)
        except (ArpackNoConvergence, Exception) as exc:
            warnings.warn(f"{name}: eigsh failed ({type(exc).__name__}); λ₁=NaN")
            lambda_1 = float("nan")

        # ── Diameter ─────────────────────────────────────────────────────
        diam = compute_approx_diameter(A_lcc, n_sweeps=diameter_sweeps)

        # ── Clustering ───────────────────────────────────────────────────
        avg_c = compute_avg_clustering(src, dst, N)

        # ── Runtime statistics ───────────────────────────────────────────
        runtimes   = np.array(npz["config_runtime"], dtype=np.float64)
        n_configs  = len(runtimes)
        rt_mean    = float(np.mean(runtimes))
        rt_std     = float(np.std(runtimes, ddof=1)) if n_configs > 1 else 0.0
        rt_cv      = rt_std / rt_mean if rt_mean > 0 else float("nan")

    except Exception:
        warnings.warn(f"{name}: skipped due to unexpected error:\n{traceback.format_exc()}")
        return None

    elapsed = time.perf_counter() - t0
    return GraphProfile(
        name=name,
        split=split,
        N=N,
        E_directed=E,
        lcc_fraction=lcc_frac,
        lambda_1=lambda_1,
        diameter=diam,
        avg_clustering=avg_c,
        runtime_mean=rt_mean,
        runtime_std=rt_std,
        runtime_cv=rt_cv,
        n_configs=n_configs,
        elapsed_s=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch loop
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    splits: List[str],
    diameter_sweeps: int,
    max_graphs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Profile all .npz files in the requested splits.

    Returns a DataFrame with one row per successfully processed graph.
    """
    # Collect all files
    all_files: List[tuple[str, str]] = []    # (path, split)
    for split in splits:
        d = _SPLIT_DIRS.get(split)
        if d is None:
            print(f"WARNING: unknown split '{split}', skipping.")
            continue
        paths = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not paths:
            print(f"WARNING: no .npz files found in {d!r}")
            continue
        for p in paths:
            all_files.append((p, split))

    if max_graphs is not None:
        all_files = all_files[:max_graphs]

    total = len(all_files)
    print(f"\nProfiling {total} graphs from splits: {splits}")
    print("─" * 70)

    profiles: List[GraphProfile] = []
    t_batch = time.perf_counter()

    for i, (path, split) in enumerate(all_files, start=1):
        name = os.path.basename(path)
        print(f"  [{i:3d}/{total}]  {name:<52}", end="", flush=True)

        prof = profile_one_graph(path, split, diameter_sweeps)
        if prof is None:
            print("  SKIPPED")
            continue

        status = "✓" if not np.isnan(prof.lambda_1) else "λ₁=NaN"
        print(
            f"  N={prof.N:>6,}  λ₁={prof.lambda_1:.4f}  "
            f"d≈{prof.diameter:>3}  CV={prof.runtime_cv:.3f}  "
            f"[{prof.elapsed_s:.2f}s]  {status}"
        )
        profiles.append(prof)

    elapsed_total = time.perf_counter() - t_batch
    print("─" * 70)
    print(f"Done. {len(profiles)}/{total} graphs profiled in {elapsed_total:.1f}s\n")

    df = pd.DataFrame([asdict(p) for p in profiles])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

_ANALYSIS_COLS = [
    "N", "lambda_1", "diameter", "avg_clustering",
    "runtime_std", "runtime_cv", "n_configs",
]

def print_spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and pretty-print the Spearman correlation matrix."""
    sub = df[_ANALYSIS_COLS].dropna()
    n   = len(sub)

    # Build correlation matrix manually to also capture p-values
    cols = _ANALYSIS_COLS
    rho_matrix = np.zeros((len(cols), len(cols)))
    p_matrix   = np.zeros_like(rho_matrix)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            rho, p = spearmanr(sub[c1], sub[c2])
            rho_matrix[i, j] = rho
            p_matrix[i, j]   = p

    rho_df = pd.DataFrame(rho_matrix, index=cols, columns=cols)

    print("=" * 72)
    print(f"  SPEARMAN RANK CORRELATION MATRIX  (n = {n} graphs)")
    print("=" * 72)
    pd.set_option("display.float_format", "{:+.3f}".format)
    pd.set_option("display.width", 120)
    print(rho_df.to_string())
    pd.reset_option("display.float_format")
    print()

    # Highlight the key correlations
    print("  KEY CORRELATIONS WITH Runtime CV (ρ, p-value)")
    print("─" * 72)
    cv_idx = cols.index("runtime_cv")
    for i, col in enumerate(cols):
        if col == "runtime_cv":
            continue
        rho = rho_matrix[i, cv_idx]
        p   = p_matrix[i, cv_idx]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {col:<22}  ρ = {rho:+.4f}   p = {p:.4f}  {sig}")
    print("=" * 72)

    return rho_df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_regression_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = "#e74c3c",
    label_prefix: str = "",
) -> None:
    """OLS regression line + shaded 95 % confidence band via numpy polyfit."""
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return
    coeffs = np.polyfit(xm, ym, 1)
    x_line = np.linspace(xm.min(), xm.max(), 300)
    y_line = np.polyval(coeffs, x_line)

    # Residual std for a rough confidence band
    y_hat   = np.polyval(coeffs, xm)
    resid   = ym - y_hat
    se      = np.std(resid) * np.sqrt(1 / len(xm) + (x_line - xm.mean()) ** 2
                                      / np.sum((xm - xm.mean()) ** 2))
    ax.plot(x_line, y_line, color=color, lw=2.0, zorder=4,
            label=f"{label_prefix}OLS fit")
    ax.fill_between(x_line, y_line - 1.96 * se, y_line + 1.96 * se,
                    color=color, alpha=0.15, zorder=3)


def _colorbar_by_logN(
    fig: plt.Figure,
    ax: plt.Axes,
    sc: plt.cm.ScalarMappable,
    label: str = r"$\log_{10}(N)$",
) -> None:
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(label, fontsize=10)


def generate_plots(df: pd.DataFrame, rho_df: pd.DataFrame) -> None:
    """Produce four publication-quality figures and save to scripts/."""
    sns.set_theme(style="whitegrid", font_scale=1.15)
    cmap = plt.cm.viridis

    # Subset to rows with valid lambda_1 and runtime_cv
    valid = df.dropna(subset=["lambda_1", "runtime_cv"]).copy()
    valid["log_N"]       = np.log10(valid["N"].astype(float))
    valid["log_lambda1"] = np.log10(valid["lambda_1"].clip(lower=1e-9))

    norm = mcolors.Normalize(vmin=valid["log_N"].min(), vmax=valid["log_N"].max())

    # ── Figure 1: Spectral Gap vs Runtime CV ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(
        valid["log_lambda1"], valid["runtime_cv"],
        c=valid["log_N"], cmap=cmap, norm=norm,
        s=65, edgecolors="white", linewidths=0.4, zorder=5, alpha=0.88,
    )
    _add_regression_line(ax, valid["log_lambda1"].values, valid["runtime_cv"].values)

    rho, pval = spearmanr(valid["lambda_1"], valid["runtime_cv"])
    ax.set_xlabel(r"Spectral Gap  $\lambda_1$  ($\log_{10}$ scale)", fontsize=12)
    ax.set_ylabel(r"Runtime Coefficient of Variation (CV = $\sigma/\mu$)", fontsize=12)
    ax.set_title(
        f"Spectral Gap vs. Runtime Variance\n"
        rf"Spearman $\rho$ = {rho:+.3f}   p = {pval:.3f}   n = {len(valid)}",
        fontsize=12,
    )

    # Annotate the 3 highest-CV points
    for _, row in valid.nlargest(3, "runtime_cv").iterrows():
        ax.annotate(
            row["name"].replace(".npz", ""),
            xy=(row["log_lambda1"], row["runtime_cv"]),
            xytext=(6, 2), textcoords="offset points",
            fontsize=7, color="#555",
        )

    _colorbar_by_logN(fig, ax, sc)
    ax.set_xticks(np.arange(
        np.floor(valid["log_lambda1"].min()),
        np.ceil(valid["log_lambda1"].max()) + 0.5, 0.5,
    ))
    ax.set_xticklabels(
        [rf"$10^{{{x:.1f}}}$" for x in ax.get_xticks()], fontsize=9
    )
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig1_spectral_gap_vs_cv.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Figure 2: Diameter vs Runtime CV ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(valid))
    sc = ax.scatter(
        valid["diameter"] + jitter, valid["runtime_cv"],
        c=valid["log_N"], cmap=cmap, norm=norm,
        s=65, edgecolors="white", linewidths=0.4, zorder=5, alpha=0.88,
    )
    _add_regression_line(ax, valid["diameter"].values.astype(float),
                         valid["runtime_cv"].values)

    rho, pval = spearmanr(valid["diameter"], valid["runtime_cv"])
    ax.set_xlabel("Approximate Graph Diameter  (hops, ±0.3 jitter)", fontsize=12)
    ax.set_ylabel(r"Runtime Coefficient of Variation (CV = $\sigma/\mu$)", fontsize=12)
    ax.set_title(
        f"Graph Diameter vs. Runtime Variance\n"
        rf"Spearman $\rho$ = {rho:+.3f}   p = {pval:.3f}   n = {len(valid)}",
        fontsize=12,
    )
    _colorbar_by_logN(fig, ax, sc)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig2_diameter_vs_cv.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Figure 3: N vs Spectral Gap (structural scaling) ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))

    c_norm = mcolors.Normalize(
        vmin=valid["avg_clustering"].min(),
        vmax=valid["avg_clustering"].quantile(0.95),   # cap at 95th pct
    )
    sc = ax.scatter(
        valid["N"], valid["lambda_1"],
        c=valid["avg_clustering"], cmap="plasma", norm=c_norm,
        s=65, edgecolors="white", linewidths=0.4, zorder=5, alpha=0.88,
    )
    _add_regression_line(
        ax,
        np.log10(valid["N"].values.astype(float)),
        np.log10(valid["lambda_1"].clip(lower=1e-9).values),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    rho, pval = spearmanr(valid["N"], valid["lambda_1"])
    ax.set_xlabel("Number of Nodes  N  (log scale)", fontsize=12)
    ax.set_ylabel(r"Spectral Gap  $\lambda_1$  (log scale)", fontsize=12)
    ax.set_title(
        "Graph Scale vs. Spectral Gap\n"
        rf"Spearman $\rho$ = {rho:+.3f}   p = {pval:.3f}   n = {len(valid)}"
        "\nColor = avg. clustering coefficient",
        fontsize=12,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Avg. Clustering Coefficient C", fontsize=10)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig3_N_vs_spectral_gap.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Figure 4: Spearman Correlation Heatmap ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6.5))
    display_cols = {
        "N":              "N",
        "lambda_1":       "lambda_1",
        "diameter":       "Diameter",
        "avg_clustering": "Clustering C",
        "runtime_mean":   "Runtime μ",
        "runtime_std":    "Runtime σ",
        "runtime_cv":     "Runtime CV",
        "n_configs":      "# Configs",
    }
    sub = df[[c for c in display_cols if c in df.columns]].dropna()
    corr_matrix = sub.corr(method="spearman")
    corr_matrix.columns = [display_cols[c] for c in corr_matrix.columns]
    corr_matrix.index   = [display_cols[c] for c in corr_matrix.index]

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)   # upper-tri mask
    sns.heatmap(
        corr_matrix,
        ax=ax,
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": r"Spearman $\rho$", "shrink": 0.8},
    )
    ax.set_title(
        f"Spearman Rank Correlation Matrix\n"
        f"(n = {len(sub)} graphs, layout:xla:random)",
        fontsize=12, pad=12,
    )
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig4_correlation_heatmap.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 0.5 – Batch topology correlation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--splits", nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Which data splits to include (default: all three)",
    )
    p.add_argument(
        "--max-graphs", type=int, default=None,
        help="Cap the number of graphs processed (useful for quick smoke tests)",
    )
    p.add_argument(
        "--diameter-sweeps", type=int, default=3,
        help="Double-sweep BFS passes for diameter (default: 3)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Batch processing ──────────────────────────────────────────────────
    df = run_batch(
        splits=args.splits,
        diameter_sweeps=args.diameter_sweeps,
        max_graphs=args.max_graphs,
    )

    if df.empty:
        sys.exit("ERROR: no graphs were successfully profiled. Check data paths.")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(SCRIPT_DIR, "topology_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved → {csv_path}")

    # ── Summary statistics ────────────────────────────────────────────────
    print("\n  SUMMARY STATISTICS")
    print("─" * 72)
    summary_cols = ["N", "lambda_1", "diameter", "avg_clustering",
                    "runtime_cv", "n_configs"]
    print(df[summary_cols].describe().map(
        lambda x: f"{x:.4f}" if isinstance(x, float) else x
    ).to_string())
    print()

    # ── Spearman correlation ──────────────────────────────────────────────
    rho_df = print_spearman_table(df)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    generate_plots(df, rho_df)

    print("\nPhase 0.5 complete.")
    print(f"  CSV    → {csv_path}")
    print(f"  Plots  → {SCRIPT_DIR}/fig{{1-4}}_*.png")


if __name__ == "__main__":
    main()
