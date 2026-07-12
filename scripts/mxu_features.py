#!/usr/bin/env python3
"""
Phase 8b — MXU-alignment feature derivation (TPU v3 systolic array co-design).

STANDALONE & SIDE-EFFECT-FREE BY DESIGN. This module does NOT modify dataset.py.
The Phase 6 expansion sweep re-imports dataset.py for each of its 30 jobs, so the
158-dim feature path must stay frozen until that sweep finishes; integration of
these features happens in Phase 8c (which retrains from scratch). Here we build
and validate the feature functions in isolation against real npz graphs.

THE HARDWARE MODEL
    TPU v3 MXU = a 128×128 systolic array (bf16). A convolution is lowered to a
    matrix multiply whose dimensions must be tiled onto 128×128 blocks. When an
    operand dimension is not a multiple of 128 the array is PADDED with zeros and
    those lanes do no useful work — wasted MXU cycles. Static graph features
    encode the raw shapes but NOT this alignment, which is the representational
    gap Phase 5 surfaced (convolution graphs ~47% off-optimal). These features
    make the alignment explicit.

FEATURES (per convolution node), all node-local & rigorously grounded:
    c_out                 output channels (GEMM N dimension)
    c_out_tiles           ceil(c_out / 128)              # MXU column tiles
    c_out_mxu_util        c_out / (c_out_tiles*128)      # ∈(0,1]; 1 = perfect fit
    c_out_pad_waste       c_out_tiles*128 - c_out        # padded-out columns
    kernel_area           kH * kW   (1 ⇒ a pure 1×1 GEMM, maximally MXU-friendly)
    gemm_m                batch * out_spatial            # GEMM M dimension (rows)
    m_mxu_util            gemm_m / (ceil(gemm_m/128)*128)
    is_grouped            feature_group_count > 1        # depthwise/grouped conv

    (The contraction dimension K = C_in*kH*kW and arithmetic intensity need the
     INPUT operand's channel count, which is not node-local; that increment is
     deferred to Phase 8c via edge_index operand lookup — see TODO at bottom.)

CONTRACTION FEATURES (added Phase 8b ext — need edge_index operand lookup):
    c_in                  total input channels (kernel C_in/group × group_count)
    contraction_k         (C_in/group)*kH*kW = the GEMM K (reduction) dimension,
                          tiled onto the MXU's 128 stationary-weight rows
    k_tiles               ceil(K / 128)
    k_mxu_util            K / (k_tiles*128)              # ∈(0,1]; row-tile fit
    k_pad_waste           k_tiles*128 - K
    n_mxu_passes          c_out_tiles * k_tiles          # 128×128 weight tiles loaded
    log10_macs            log10 of M*K*N multiply-accumulates (compute scale)
    arithmetic_intensity  MACs / (input+weight+output elems)  # roofline proxy

EDGE CONVENTION (verified on ResNet): edge_index rows are [consumer, operand];
a conv's operands are edge_index[edge_index[:,0]==conv, 1]. The KERNEL operand is
identified self-consistently: its shape at the conv's kernel_spatial axes == kH,kW
AND at the kernel_output_feature axis == C_out; C_in is then read at the
kernel_input_feature axis. Conv nodes whose kernel can't be resolved this way get
NaN contraction features (counted, not silently zeroed).

USAGE
    from scripts.mxu_features import compute_mxu_features
    feats, names = compute_mxu_features(node_feat, node_opcode)              # node-local only
    feats, names = compute_mxu_features(node_feat, node_opcode, edge_index)  # + contraction

    python scripts/mxu_features.py            # validate on a ResNet npz, show stats
    python scripts/mxu_features.py PATH.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

CONV_OPCODE = 26
MXU_DIM = 128  # TPU v3 systolic array edge length

_IDX_PATH = Path("experiments/conv_feature_indices.json")

# Output feature names, fixed order (must match the columns built below).
# First block = node-local; second block = contraction (needs edge_index).
NODE_LOCAL_NAMES = [
    "c_out",
    "c_out_tiles",
    "c_out_mxu_util",
    "c_out_pad_waste",
    "kernel_area",
    "gemm_m",
    "m_mxu_util",
    "is_grouped",
]
CONTRACTION_NAMES = [
    "c_in",
    "contraction_k",
    "k_tiles",
    "k_mxu_util",
    "k_pad_waste",
    "n_mxu_passes",
    "log10_macs",
    "arithmetic_intensity",
]
FEATURE_NAMES = NODE_LOCAL_NAMES + CONTRACTION_NAMES


def _load_indices() -> dict:
    """Column indices from the Phase 8a schema (experiments/node_feature_schema.json),
    falling back to conv_feature_indices.json for the MXU subset."""
    schema_path = Path("experiments/node_feature_schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(
            f"{schema_path} missing — run scripts/extract_conv_features.py first."
        )
    name2idx = {v: int(k) for k, v in
                json.loads(schema_path.read_text())["index_to_name"].items()}
    cdn = "convolution_dim_numbers"
    return {
        "kernel_h": name2idx["window_size_0"],
        "kernel_w": name2idx["window_size_1"],
        # ALL window slots — 3-D convs (e.g. unet_3d) use window_size_2; reading
        # only 0/1 undercounts kernel volume (3×3×3 → 9 instead of 27).
        "kernel_dims": [name2idx[f"window_size_{i}"] for i in range(6)],
        "stride_h": name2idx["window_stride_0"],
        "stride_w": name2idx["window_stride_1"],
        "stride_dims": [name2idx[f"window_stride_{i}"] for i in range(6)],
        "shape_dims": [name2idx[f"shape_dimensions_{i}"] for i in range(6)],
        "out_feature_axis": name2idx[f"{cdn}_output_feature_dim"],
        "kernel_in_axis": name2idx[f"{cdn}_kernel_input_feature_dim"],
        "kernel_out_axis": name2idx[f"{cdn}_kernel_output_feature_dim"],
        "kernel_spatial_axes": [name2idx[f"{cdn}_kernel_spatial_dims_{i}"] for i in range(4)],
        "feature_group_count": name2idx["feature_group_count"],
    }


def _ceil_div(a: np.ndarray, b: int) -> np.ndarray:
    return -(-a // b)


def _resolve_c_in(
    conv_nodes: np.ndarray,
    node_feat: np.ndarray,
    edge_index: np.ndarray,
    idx: dict,
    c_out: np.ndarray,
    kh: np.ndarray,
    kw: np.ndarray,
) -> np.ndarray:
    """Per conv node, recover input channels C_in (total) via operand lookup.

    Edge convention [consumer, operand]: operands = edge_index[edge_index[:,0]==c, 1].
    The kernel operand is the one whose shape at the conv's kernel-spatial axes
    matches (kH,kW) AND at the kernel-output axis matches C_out; C_in is read at
    the kernel-input axis (× feature_group_count for grouped convs). Returns NaN
    where no operand satisfies the self-consistency check.
    """
    shp = np.asarray(idx["shape_dims"])
    # adjacency: consumer -> list of operands
    operands: dict = {}
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    for c in conv_nodes:
        operands[c] = dst[src == c]

    c_in = np.full(len(conv_nodes), np.nan)
    for i, c in enumerate(conv_nodes):
        f = node_feat[c]
        kin_ax = int(f[idx["kernel_in_axis"]])
        kout_ax = int(f[idx["kernel_out_axis"]])
        ksp = [int(f[a]) for a in idx["kernel_spatial_axes"][:2]]
        fgc = max(int(f[idx["feature_group_count"]]), 1)
        khw = (int(kh[i]), int(kw[i]))
        for op_node in operands[c]:
            kshape = node_feat[op_node][shp].astype(int)
            if not (0 <= kin_ax < 6 and 0 <= kout_ax < 6):
                continue
            # ksp holds the conv's kernel-spatial AXIS NUMBERS (e.g. [0,1]); index
            # the operand shape by those, not by the schema column positions.
            sp = tuple(int(kshape[a]) for a in ksp if 0 <= a < 6)
            # self-consistency: spatial==kHkW and output-feature axis==C_out
            if sp == khw and int(kshape[kout_ax]) == int(c_out[i]):
                c_in[i] = float(kshape[kin_ax]) * fgc
                break
    return c_in


def compute_mxu_features(
    node_feat: np.ndarray,
    node_opcode: np.ndarray,
    edge_index: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return (N, len(FEATURE_NAMES)) float32 MXU features; non-conv rows are 0.

    node_feat   : (N, 140) RAW node features (pre log1p — sizes are literal ints)
    node_opcode : (N,) opcode ids
    edge_index  : (E, 2) [consumer, operand]; if given, the contraction features
                  (c_in, contraction_k, k_*, macs, arithmetic_intensity) are
                  filled. Conv nodes whose kernel can't be resolved get NaN there.
    """
    idx = _load_indices()
    N = node_feat.shape[0]
    out = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)

    conv = np.where(node_opcode == CONV_OPCODE)[0]
    if conv.size == 0:
        return out, FEATURE_NAMES

    feat = np.asarray(node_feat[conv], dtype=np.float64)

    # ── C_out: the output-shape extent on the feature axis ────────────────────
    shape_cols = np.array(idx["shape_dims"])               # 6 shape cols
    feat_axis = np.clip(feat[:, idx["out_feature_axis"]].astype(int), 0, 5)
    out_shape = feat[:, shape_cols]                        # (n_conv, 6)
    c_out = np.maximum(out_shape[np.arange(len(conv)), feat_axis], 1.0)

    # ── GEMM M = batch * spatial output positions (all output dims except C) ──
    prod_all = np.prod(np.where(out_shape > 0, out_shape, 1.0), axis=1)
    gemm_m = np.maximum(prod_all / c_out, 1.0)

    kh = np.maximum(feat[:, idx["kernel_h"]], 1.0)
    kw = np.maximum(feat[:, idx["kernel_w"]], 1.0)
    # Kernel VOLUME over all spatial dims (window_size_0..5; zeros = absent dims).
    # 2-D convs: kH·kW unchanged. 3-D convs (unet_3d): kD·kH·kW — the 2-slot
    # version undercounted K/MACs/AI by the kernel depth.
    k_vol = np.prod(np.maximum(feat[:, idx["kernel_dims"]], 1.0), axis=1)
    s_vol = np.prod(np.maximum(feat[:, idx["stride_dims"]], 1.0), axis=1)
    fgc = feat[:, idx["feature_group_count"]]

    c_out_tiles = _ceil_div(c_out.astype(np.int64), MXU_DIM).astype(np.float64)
    m_tiles = _ceil_div(gemm_m.astype(np.int64), MXU_DIM).astype(np.float64)

    node_local = np.stack([
        c_out,
        c_out_tiles,
        c_out / (c_out_tiles * MXU_DIM),
        c_out_tiles * MXU_DIM - c_out,
        k_vol,                       # kernel_area = full spatial volume (2-D: kH·kW; 3-D: kD·kH·kW)
        gemm_m,
        gemm_m / (m_tiles * MXU_DIM),
        (fgc > 1).astype(np.float64),
    ], axis=1)
    out[conv, :len(NODE_LOCAL_NAMES)] = node_local.astype(np.float32)

    # ── Contraction features (need operand lookup) ────────────────────────────
    if edge_index is not None:
        ei = np.asarray(edge_index)
        c_in = _resolve_c_in(conv, np.asarray(node_feat, dtype=np.float64),
                             ei, idx, c_out, kh, kw)
        # per-group input channels = c_in / groups; contraction K = (C_in/g)·kernel_volume
        cin_per_g = c_in / np.maximum(fgc, 1.0)
        K = cin_per_g * k_vol                              # GEMM reduction dim
        k_tiles = _ceil_div(np.nan_to_num(K, nan=1.0).astype(np.int64),
                            MXU_DIM).astype(np.float64)
        macs = gemm_m * K * c_out                          # M·K·N
        # roofline proxy: AI = MACs / (input + weight + output) element counts
        weight_el = K * c_out
        output_el = gemm_m * c_out
        input_el = gemm_m * s_vol * c_in                   # ≈ batch·(spatial_in)·C_in
        ai = macs / (input_el + weight_el + output_el)

        contraction = np.stack([
            c_in,
            K,
            k_tiles,
            K / (k_tiles * MXU_DIM),
            k_tiles * MXU_DIM - K,
            c_out_tiles * k_tiles,
            np.log10(np.maximum(macs, 1.0)),
            ai,
        ], axis=1)
        out[conv, len(NODE_LOCAL_NAMES):] = contraction.astype(np.float32)

    return out, FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Validation / demo
# ─────────────────────────────────────────────────────────────────────────────
def _demo(npz_path: str) -> None:
    npz = np.load(npz_path, mmap_mode="r")
    node_feat = np.asarray(npz["node_feat"])
    node_opcode = np.asarray(npz["node_opcode"])
    edge_index = np.asarray(npz["edge_index"])
    feats, names = compute_mxu_features(node_feat, node_opcode, edge_index)

    conv = np.where(node_opcode == CONV_OPCODE)[0]
    print(f"file: {Path(npz_path).name}   conv nodes: {len(conv)}")
    if len(conv) == 0:
        return

    ci = {n: i for i, n in enumerate(names)}
    print("\n  sample conv nodes (channel + contraction MXU alignment):")
    print(f"  {'node':>6} {'C_in':>5} {'C_out':>6} {'kernel':>7} "
          f"{'K=cin·kk':>9} {'k_util':>7} {'cout_util':>9} {'log10MACs':>9} {'AI':>6}")
    for nid in conv[:10]:
        r = feats[nid]
        cin = r[ci['c_in']]
        print(f"  {int(nid):>6} {('?' if np.isnan(cin) else int(cin)):>5} "
              f"{int(r[ci['c_out']]):>6} {int(r[ci['kernel_area']]):>7} "
              f"{('?' if np.isnan(r[ci['contraction_k']]) else int(r[ci['contraction_k']])):>9} "
              f"{r[ci['k_mxu_util']]:>7.3f} {r[ci['c_out_mxu_util']]:>9.3f} "
              f"{r[ci['log10_macs']]:>9.2f} {r[ci['arithmetic_intensity']]:>6.1f}")

    util = feats[conv, ci["c_out_mxu_util"]]
    cin = feats[conv, ci["c_in"]]
    kutil = feats[conv, ci["k_mxu_util"]]
    resolved = ~np.isnan(cin)
    print(f"\n  ── C_in operand resolution ──")
    print(f"  resolved {resolved.sum()}/{len(conv)} conv nodes "
          f"({resolved.mean()*100:.0f}%) via self-consistent kernel match.")
    print(f"\n  ── MXU systolic-array utilisation (the co-design signal) ──")
    print(f"  C_out (columns): mean {util.mean():.3f}  min {util.min():.3f}  "
          f"<0.75 in {(util<0.75).mean()*100:.0f}% of nodes")
    if resolved.any():
        ku = kutil[resolved]
        print(f"  K=C_in·kH·kW (rows): mean {ku.mean():.3f}  min {ku.min():.3f}  "
              f"<0.75 in {(ku<0.75).mean()*100:.0f}% of resolved nodes")
    print("  ↳ convolution layers whose channel/contraction dims misalign to the")
    print("    128×128 MXU waste array cycles — invisible to static features (Phase 5 gap).")


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import glob
        cands = glob.glob("data/predict-ai-model-runtime/npz_all/npz/layout/"
                          "xla/*/*/inference_mlperf_resnet*.npz")
        if not cands:
            print("no ResNet npz found; pass a path explicitly.", file=sys.stderr)
            return 1
        path = cands[0]
    _demo(path)
    return 0


# TODO(Phase 8c): contraction-dim features need C_in = input-activation channel
# count, recovered via edge_index operand lookup (the conv's data predecessor).
# Add: K = C_in*kH*kW, k_mxu_util = K/(ceil(K/128)*128), n_mxu_passes =
# c_out_tiles*ceil(K/128), arithmetic_intensity = MACs/bytes (roofline). Validate
# C_in against a known ResNet block (e.g. 3×3 conv with C_in=64) before trusting.

if __name__ == "__main__":
    sys.exit(main())
