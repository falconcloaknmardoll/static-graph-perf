#!/usr/bin/env python3
"""
analyze_opcodes.py
------------------
Days 11 & 12 – Opcode & Config-Density Deep-Dive.

For each of up to N_GRAPHS graphs from layout:xla:random/train, computes:

  1. Compute Ratio
     Fraction of nodes whose XLA opcode is compute-bound (vs memory-bound).
     Memory-bound ops: data movement, reshaping, indexing, communication.
     Compute-bound ops: arithmetic, convolutions, reductions, activations.

  2. Config Spread  (Configurable-Node Density)
     Median pairwise shortest-path distance (in hops, undirected graph)
     between the K nodes the compiler is allowed to modify (node_config_ids).
     Low spread (1–2) → all config nodes are clustered.
     High spread         → config nodes are dispersed across the full graph.
     Normalized spread   = median_config_dist / approx_diameter  ∈ [0, 1].

  3. runtime_std
     Standard deviation of config_runtime across all compiler configurations.
     Measures how much the compiler's choices actually affect latency.

Correlations (Spearman ρ) of metrics 1 & 2 against metric 3 are reported.

Usage
-----
    python analyze_opcodes.py                     # 100 graphs, verbose
    python analyze_opcodes.py --n-graphs 50
    python analyze_opcodes.py --split valid
    python analyze_opcodes.py --no-plot           # skip matplotlib figures
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------------------------
# XLA HloOpcode categorisation
# ---------------------------------------------------------------------------
# Source: XLA HLO op enum (alphabetical, 0-indexed).
# Ref: tensorflow/compiler/xla/hlo/ir/hlo_opcode.h
#
# Memory-bound: data movement, layout change, communication, I/O.
# Compute-bound: arithmetic, activation, reduction, math functions.
# Other (infrastructure): async handles, barriers, control flow stubs.
# ---------------------------------------------------------------------------

MEMORY_BOUND_OPCODES: Set[int] = {
    4, 5, 6,          # AllGather, AllGatherDone, AllGatherStart
    7, 8, 9,          # AllReduce, AllReduceDone, AllReduceStart
    10,               # AllToAll
    16, 17,           # Bitcast, BitcastConvert
    18,               # Broadcast
    22, 23, 24,       # CollectivePermute, CollectivePermuteDone, CollectivePermuteStart
    27,               # Concatenate
    28,               # Constant
    29,               # Convert (type cast – data movement dominated)
    31, 32, 33,       # Copy, CopyDone, CopyStart
    40, 41,           # DynamicSlice, DynamicUpdateSlice
    47,               # Gather
    48,               # GetDimensionSize
    49,               # GetTupleElement
    51,               # Infeed
    52,               # Iota
    68,               # Outfeed
    69,               # Pad
    70,               # Parameter
    75, 76,           # Recv, RecvDone
    79,               # ReduceScatter (communication)
    83,               # Reshape
    84,               # Reverse
    91,               # Scatter
    94, 95,           # Send, SendDone
    96,               # SetDimensionSize
    102,              # Slice
    107,              # Transpose
    108,              # Tuple
}

COMPUTE_BOUND_OPCODES: Set[int] = {
    0,                # Abs
    1,                # Add
    13, 14, 15,       # BatchNormGrad, BatchNormInference, BatchNormTraining
    20,               # Cholesky
    21,               # Clamp
    25,               # Compare
    26,               # Complex
    30,               # Convolution
    34,               # CountLeadingZeros
    37,               # Divide
    39,               # Dot
    42, 43,           # Exponential, ExponentialMinusOne
    44,               # Fft
    45,               # Floor
    46,               # Fusion
    50,               # Imag
    54, 55,           # Log, LogPlusOne
    56, 57, 58,       # LogicalAnd, LogicalNot, LogicalOr
    59,               # Logistic (sigmoid)
    60,               # Map
    61, 62,           # Maximum, Minimum
    63,               # Multiply
    64,               # Negate
    65, 67,           # Not, Or
    72,               # PopulationCount
    73,               # Power
    74,               # Real
    77,               # Reduce
    78,               # ReducePrecision
    80,               # ReduceWindow
    81,               # Remainder
    85, 86, 87,       # Rng, RngBitGenerator, RngGetAndUpdateState
    88, 89,           # RoundNearestAfz, RoundNearestEven
    90,               # Rsqrt
    92, 93,           # Select, SelectAndScatter
    97, 98, 99,       # ShiftLeft, ShiftRightArithmetic, ShiftRightLogical
    100,              # Sign
    101,              # Sin
    103,              # Sort
    104,              # Sqrt
    105,              # Subtract
    106,              # Tanh
    109,              # TupleReduce
    110,              # While
    111,              # Xor
}

# Opcodes outside both sets (2=AddDependency, 3=AfterAll, 11/12=Async*, 19=Call,
# 35=CrossReplicaSum, 36=CustomCall, 38=Domain, 53=IsFinite,
# 66=OptimizationBarrier, 71=PartitionId, 82=ReplicaId) fall in "other".


def classify_opcode(op: int) -> str:
    if op in COMPUTE_BOUND_OPCODES:
        return "compute"
    if op in MEMORY_BOUND_OPCODES:
        return "memory"
    return "other"


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def build_adjacency_list(edge_index: np.ndarray, num_nodes: int) -> List[List[int]]:
    """Undirected adjacency list from PyG (2, E) edge_index."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0], edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def bfs_distances(adj: List[List[int]], source: int) -> Dict[int, int]:
    """BFS from source; returns {node: distance}."""
    dist: Dict[int, int] = {source: 0}
    q: deque[int] = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def approx_diameter(adj: List[List[int]], n_sweeps: int = 3, seed: int = 0) -> int:
    """Lower-bound diameter via iterated double-sweep BFS."""
    rng = np.random.default_rng(seed)
    nodes = list(range(len(adj)))
    best = 0
    for _ in range(n_sweeps):
        u = int(rng.choice(nodes))
        d1 = bfs_distances(adj, u)
        v = max(d1, key=d1.get)
        d2 = bfs_distances(adj, v)
        best = max(best, max(d2.values()))
    return best


def compute_config_spread(
    adj: List[List[int]],
    config_ids: np.ndarray,
    max_pairs: int = 300,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute median pairwise hop-distance between configurable nodes.

    For K config nodes, there are K*(K-1)/2 pairs; we sample up to
    `max_pairs` of them to keep runtime bounded on large graphs.

    Returns
    -------
    median_dist  : median pairwise shortest-path distance (hops)
    fraction_close : fraction of sampled pairs within 2 hops (dense proxy)
    """
    K = len(config_ids)
    if K < 2:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    cids = config_ids.tolist()

    # Sample source nodes; run BFS from each to get distances to all other
    # config nodes.  Stop early if we have enough pairs.
    all_dists: List[int] = []
    sampled_sources: List[int] = cids[:]
    if K > max_pairs:
        sampled_sources = rng.choice(cids, size=max_pairs, replace=False).tolist()

    config_id_set: Set[int] = set(cids)
    for src in sampled_sources:
        d = bfs_distances(adj, src)
        for tgt in cids:
            if tgt != src:
                all_dists.append(d.get(tgt, 10_000))
        if len(all_dists) >= max_pairs * 10:
            break

    if not all_dists:
        return 0.0, 1.0

    arr = np.array(all_dists, dtype=np.float32)
    median_d = float(np.median(arr))
    frac_close = float(np.mean(arr <= 2))
    return median_d, frac_close


# ---------------------------------------------------------------------------
# Per-graph analysis
# ---------------------------------------------------------------------------

@dataclass
class GraphMetrics:
    filename: str
    n_nodes: int
    n_config_nodes: int
    n_configs: int
    compute_ratio: float       # fraction of compute-bound nodes
    memory_ratio: float        # fraction of memory-bound nodes
    other_ratio: float         # fraction of "other" (infrastructure) nodes
    median_config_dist: float  # median pairwise config-node distance (hops)
    fraction_close: float      # fraction of config-node pairs within 2 hops
    norm_config_spread: float  # median_config_dist / approx_diameter
    approx_diam: int
    runtime_std: float
    runtime_mean: float
    runtime_cv: float          # coefficient of variation


def analyze_graph(path: str, n_diameter_sweeps: int = 3) -> Optional[GraphMetrics]:
    """Analyse a single .npz file; return None on error."""
    try:
        npz = np.load(path, mmap_mode="r")
        opcodes     = np.array(npz["node_opcode"],      dtype=np.int32)
        edge_index  = np.array(npz["edge_index"].T,     dtype=np.int64)   # (2, E)
        config_ids  = np.array(npz["node_config_ids"],  dtype=np.int64)
        runtimes    = np.array(npz["config_runtime"],   dtype=np.float64)
        N = opcodes.shape[0]

        # ── 1. Compute / Memory ratio ──────────────────────────────────────
        labels = np.array([classify_opcode(int(op)) for op in opcodes])
        n_compute = int(np.sum(labels == "compute"))
        n_memory  = int(np.sum(labels == "memory"))
        n_other   = int(np.sum(labels == "other"))

        compute_ratio = n_compute / N
        memory_ratio  = n_memory  / N
        other_ratio   = n_other   / N

        # ── 2. Config spread ───────────────────────────────────────────────
        adj = build_adjacency_list(edge_index, N)
        diam = approx_diameter(adj, n_sweeps=n_diameter_sweeps)
        med_dist, frac_close = compute_config_spread(adj, config_ids)
        norm_spread = med_dist / max(diam, 1)

        # ── 3. Runtime stats ───────────────────────────────────────────────
        rt_std  = float(np.std(runtimes))
        rt_mean = float(np.mean(runtimes))
        rt_cv   = rt_std / rt_mean if rt_mean > 0 else 0.0

        return GraphMetrics(
            filename=os.path.basename(path),
            n_nodes=N,
            n_config_nodes=int(len(config_ids)),
            n_configs=int(len(runtimes)),
            compute_ratio=compute_ratio,
            memory_ratio=memory_ratio,
            other_ratio=other_ratio,
            median_config_dist=med_dist,
            fraction_close=frac_close,
            norm_config_spread=norm_spread,
            approx_diam=diam,
            runtime_std=rt_std,
            runtime_mean=rt_mean,
            runtime_cv=rt_cv,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] {os.path.basename(path)}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Correlation & reporting
# ---------------------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def print_report(records: List[GraphMetrics]) -> None:
    W = 72
    sep  = "═" * W
    thin = "─" * W

    def row(label: str, val: str) -> None:
        print(f"  {label:<44}{val}")

    print(f"\n{sep}")
    print(f"  TPUGraphs — Opcode & Config Density Analysis")
    print(f"  Days 11 & 12  |  N = {len(records)} graphs")
    print(sep)

    # ── Dataset-level summaries ───────────────────────────────────────────
    cr  = np.array([r.compute_ratio        for r in records])
    mr  = np.array([r.memory_ratio         for r in records])
    or_ = np.array([r.other_ratio          for r in records])
    mcd = np.array([r.median_config_dist   for r in records])
    fc  = np.array([r.fraction_close       for r in records])
    ns  = np.array([r.norm_config_spread   for r in records])
    rs  = np.array([r.runtime_std          for r in records])
    rv  = np.array([r.runtime_cv           for r in records])
    nn  = np.array([r.n_nodes              for r in records])

    print(f"\n  1. COMPUTE vs MEMORY RATIO (across {len(records)} graphs)")
    print(thin)
    row("Mean compute-bound fraction",  f"{cr.mean():.3f}  ± {cr.std():.3f}")
    row("Mean memory-bound fraction",   f"{mr.mean():.3f}  ± {mr.std():.3f}")
    row("Mean infrastructure fraction", f"{or_.mean():.3f}  ± {or_.std():.3f}")
    row("Compute ratio range",          f"[{cr.min():.3f}, {cr.max():.3f}]")
    row("Memory  ratio range",          f"[{mr.min():.3f}, {mr.max():.3f}]")

    # Dominant class
    dom = "COMPUTE-dominated" if cr.mean() > mr.mean() else "MEMORY-dominated"
    print(f"\n  ↳ These XLA graphs are on average {dom}")
    print(f"    (compute:{cr.mean():.1%}  memory:{mr.mean():.1%}  other:{or_.mean():.1%})")

    print(f"\n  2. CONFIGURABLE NODE DENSITY")
    print(thin)
    row("Median pairwise config-dist (hops)", f"{mcd.mean():.1f}  ± {mcd.std():.1f}")
    row("Fraction of config pairs ≤2 hops",  f"{fc.mean():.3f}  ± {fc.std():.3f}")
    row("Normalized spread (dist/diameter)",  f"{ns.mean():.3f}  ± {ns.std():.3f}")
    row("Config dist range",                  f"[{mcd.min():.1f}, {mcd.max():.1f}]")

    if mcd.mean() <= 2.0:
        density_verdict = "DENSE — config nodes are tightly clustered (≤2 hops apart on average)"
    elif mcd.mean() <= 5.0:
        density_verdict = "MODERATE — config nodes form a loose neighbourhood"
    else:
        density_verdict = "SPARSE — config nodes are dispersed across the full graph"
    print(f"\n  ↳ {density_verdict}")

    print(f"\n  3. RUNTIME VARIANCE SUMMARY")
    print(thin)
    row("Mean runtime_std",  f"{rs.mean():.3e}  ± {rs.std():.3e}")
    row("Mean runtime_cv",   f"{rv.mean():.4f}  ± {rv.std():.4f}")
    row("Graphs w/ cv > 0.1",
        f"{int(np.sum(rv > 0.1))} / {len(records)}  ({np.mean(rv > 0.1):.0%})")

    # ── Correlations ──────────────────────────────────────────────────────
    print(f"\n  4. SPEARMAN CORRELATIONS with runtime_std")
    print(thin)

    pairs = [
        ("compute_ratio",      cr,  "Compute-bound fraction of nodes"),
        ("memory_ratio",       mr,  "Memory-bound fraction of nodes"),
        ("median_config_dist", mcd, "Median pairwise config-node distance"),
        ("fraction_close",     fc,  "Fraction of config pairs ≤2 hops"),
        ("norm_config_spread", ns,  "Normalized config spread (dist/diam)"),
        ("n_nodes",            nn,  "Graph size (num nodes)"),
    ]

    print(f"  {'Metric':<30} {'ρ':>8}  {'p-value':>10}  Interpretation")
    print(f"  {'─'*30} {'─'*8}  {'─'*10}  {'─'*24}")
    for name, x, desc in pairs:
        rho, p = spearman(x, rs)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        direction = "+" if rho > 0 else "-"
        strength  = (
            "negligible" if abs(rho) < 0.10 else
            "weak"       if abs(rho) < 0.25 else
            "moderate"   if abs(rho) < 0.50 else
            "strong"
        )
        print(f"  {name:<30} {rho:>+8.3f}  {p:>10.4f}  {sig:3s} {direction} {strength}")

    # ── Interpretation ────────────────────────────────────────────────────
    rho_cr, _ = spearman(cr, rs)
    rho_ns, _ = spearman(ns, rs)

    print(f"\n  5. INTERPRETATION")
    print(thin)

    if abs(rho_cr) >= 0.25:
        if rho_cr > 0:
            print(f"  • Compute ratio ρ={rho_cr:+.3f}: graphs with MORE compute ops")
            print(f"    show HIGHER runtime variance. The compiler's tiling/layout")
            print(f"    decisions interact more strongly with compute bottlenecks.")
        else:
            print(f"  • Compute ratio ρ={rho_cr:+.3f}: graphs with MORE compute ops")
            print(f"    show LOWER runtime variance. Memory ops may be the dominant")
            print(f"    variable-latency source (bandwidth unpredictability).")
    else:
        print(f"  • Compute ratio ρ={rho_cr:+.3f}: weak correlation — opcode mix alone")
        print(f"    is not a strong predictor of runtime variance.")

    if abs(rho_ns) >= 0.25:
        if rho_ns > 0:
            print(f"  • Config spread ρ={rho_ns:+.3f}: graphs with MORE DISPERSED config")
            print(f"    nodes have higher runtime variance. Compiler choices affect")
            print(f"    a wider cross-section of the data flow → more impact.")
        else:
            print(f"  • Config spread ρ={rho_ns:+.3f}: graphs with MORE CLUSTERED config")
            print(f"    nodes have higher runtime variance. Dense config regions are")
            print(f"    local bottlenecks with disproportionate impact.")
    else:
        print(f"  • Config spread ρ={rho_ns:+.3f}: weak correlation — config topology")
        print(f"    alone does not strongly predict runtime variance.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------

def save_plots(records: List[GraphMetrics], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("[INFO] matplotlib/seaborn/pandas not available — skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Days 11–12 | Opcode & Config Density vs Runtime Variance")

    # --- Compute ratio vs runtime_std ---
    axes[0].scatter(df["compute_ratio"], df["runtime_std"], alpha=0.7, s=40)
    axes[0].set_xlabel("Compute-Bound Node Fraction")
    axes[0].set_ylabel("runtime_std")
    axes[0].set_title("Compute Ratio vs Runtime Variance")
    rho, p = stats.spearmanr(df["compute_ratio"], df["runtime_std"])
    axes[0].set_title(f"Compute Ratio vs Runtime Std\nρ={rho:+.3f}  p={p:.3f}")

    # --- Config spread vs runtime_std ---
    axes[1].scatter(df["norm_config_spread"], df["runtime_std"], alpha=0.7, s=40, color="darkorange")
    rho2, p2 = stats.spearmanr(df["norm_config_spread"], df["runtime_std"])
    axes[1].set_xlabel("Normalized Config Spread (dist/diam)")
    axes[1].set_ylabel("runtime_std")
    axes[1].set_title(f"Config Spread vs Runtime Std\nρ={rho2:+.3f}  p={p2:.3f}")

    # --- Correlation heatmap ---
    cols = ["compute_ratio", "memory_ratio", "median_config_dist",
            "fraction_close", "norm_config_spread", "n_nodes", "runtime_std", "runtime_cv"]
    corr = df[cols].corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[2], vmin=-1, vmax=1)
    axes[2].set_title("Spearman ρ Heatmap")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "opcode_density_correlations.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(
    "data", "predict-ai-model-runtime",
    "npz_all", "npz", "layout", "xla", "random",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Days 11 & 12 — Opcode & Config-Density Deep-Dive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-graphs",        type=int, default=100,
                   help="Max graphs to analyse (default: 100)")
    p.add_argument("--split",           type=str, default="train",
                   choices=["train", "valid", "test"],
                   help="Dataset split (default: train)")
    p.add_argument("--diameter-sweeps", type=int, default=3,
                   help="BFS sweeps for diameter estimate (default: 3)")
    p.add_argument("--no-plot",         action="store_true",
                   help="Skip saving matplotlib figures")
    p.add_argument("--plot-dir",        type=str, default="scripts",
                   help="Output directory for plots (default: scripts/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    split_dir = os.path.join(DATA_ROOT, args.split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if not files:
        sys.exit(f"ERROR: No .npz files found under {split_dir!r}")

    files = files[: args.n_graphs]
    print(f"\nAnalysing {len(files)} graphs from {split_dir}")
    print("─" * 60)

    records: List[GraphMetrics] = []
    t0 = time.perf_counter()

    for i, path in enumerate(files, 1):
        fname = os.path.basename(path)
        print(f"  [{i:>3}/{len(files)}] {fname[:55]:<55}", end="", flush=True)
        ts = time.perf_counter()
        m = analyze_graph(path, n_diameter_sweeps=args.diameter_sweeps)
        elapsed = time.perf_counter() - ts
        if m is None:
            print(" SKIP")
            continue
        records.append(m)
        print(
            f" cr={m.compute_ratio:.2f}  spread={m.median_config_dist:.1f}  "
            f"rt_cv={m.runtime_cv:.3f}  ({elapsed:.1f}s)"
        )

    total = time.perf_counter() - t0
    print(f"\nCompleted {len(records)} graphs in {total:.1f}s")

    if len(records) < 3:
        sys.exit("ERROR: Too few valid graphs to compute correlations.")

    print_report(records)

    if not args.no_plot:
        save_plots(records, args.plot_dir)


if __name__ == "__main__":
    main()
