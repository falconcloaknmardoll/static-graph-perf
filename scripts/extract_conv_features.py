#!/usr/bin/env python3
"""
Phase 8a — Node-feature index → name reconstruction (gating task for MXU co-design).

WHAT
    Recovers the human-readable name of every column of the TPUGraphs 140-dim
    `node_feat` matrix by *parsing the Google reference featurizer headers* and
    emulating the `NamesFor*` dispatch they generate — then isolates the columns
    that describe CONVOLUTION geometry (input/kernel/output feature dims, spatial
    dims, window stride/padding/dilation, group counts). Those columns are the
    raw material for the Phase 8b MXU-alignment features (channel-128 padding
    waste, MXU utilisation, arithmetic intensity, spatial-tile→pass count).

WHY THIS IS TRUSTWORTHY (not AI-asserted — cf. VERIFICATION.md discipline)
    The reference defines, in `hlo_opcode.h`, the exact node-feature counts:
        kMinimalNodeFeatureCount        = 113
        kOpLevelNonZeroNodeFeatureCount =  27   (added for the layout task)
        => 113 + 27 = 140
    This script reconstructs the schema from the headers and then ASSERTS those
    sub-totals and the 140 total. If the parse drifts, it fails loudly rather
    than emitting a plausible-but-wrong mapping. With --validate it additionally
    cross-checks the convolution columns against a real ResNet npz (a conv node
    must carry integer kernel/spatial extents like 7,7 / 3,3 / 1,1).

THE TASK MASK
    `GetIncludeFeatureBits("module_layout_cost")` == FEATURE_OP_NON_ZERO, so the
    layout dataset uses the 113 "minimal" features + the 27 "op-non-zero" block.
    FEATURE_WINDOW / FEATURE_OP_ZERO / FEATURE_MODULE_NON_ZERO are OFF.

USAGE
    python scripts/extract_conv_features.py
    python scripts/extract_conv_features.py --task module_layout_cost
    python scripts/extract_conv_features.py --validate \
        data/predict-ai-model-runtime/npz_all/npz/layout/xla/default/train/inference_mlperf_resnet_batch_16.npz

OUTPUT
    experiments/node_feature_schema.json   full 140-col index→name table
    experiments/conv_feature_indices.json  convolution-relevant subset (Phase 8b input)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── Reference header locations ────────────────────────────────────────────────
REF = Path("reference/tpu_graphs/process_data/xla")
FEATURIZERS_H = REF / "featurizers.h"
HLO_OPCODE_H = REF / "hlo_opcode.h"

OUT_SCHEMA = Path("experiments/node_feature_schema.json")
OUT_CONV = Path("experiments/conv_feature_indices.json")

# Root featurizer that assembles the per-node vector.
ROOT_FEATURIZER = "FeaturizeHloInstruction"

# TPUGraphs node_opcode enum (custom; conv=26) — see reference/.../hlo_opcode.h.
CONV_OPCODE = 26


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parse hlo_opcode.h: FEATURE_* bit values, task→mask, declared counts
# ─────────────────────────────────────────────────────────────────────────────
def parse_hlo_opcode(path: Path) -> dict:
    text = path.read_text()

    bits: Dict[str, int] = {}
    # e.g.  #define FEATURE_WINDOW 1 << 0
    for name, shift in re.findall(r"#define\s+(FEATURE_\w+)\s+1\s*<<\s*(\d+)", text):
        bits[name] = 1 << int(shift)

    counts: Dict[str, int] = {}
    for name, val in re.findall(
        r"constexpr\s+uint16_t\s+(k\w*NodeFeatureCount)\s*=\s*(\d+)", text
    ):
        counts[name] = int(val)

    # GetIncludeFeatureBits(task): map each task string to its OR-ed flags.
    task_mask: Dict[str, int] = {}
    body = re.search(r"GetIncludeFeatureBits\([^)]*\)\s*\{(.*?)\n\}", text, re.S)
    if body:
        for cond, ret in re.findall(
            r'if\s*\((.*?)\)\s*\{?\s*return\s+([^;]+);', body.group(1), re.S
        ):
            tasks = re.findall(r'"([^"]+)"', cond)
            flags = re.findall(r"FEATURE_\w+", ret)
            mask = 0
            for fl in flags:
                mask |= bits.get(fl, 0)
            for t in tasks:
                task_mask[t] = mask

    return {"bits": bits, "counts": counts, "task_mask": task_mask}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parse featurizers.h: primitive-type names + each featurizer body
# ─────────────────────────────────────────────────────────────────────────────
def parse_featurizers(path: Path) -> Tuple[List[str], Dict[str, str]]:
    text = path.read_text()
    # Strip // line comments (kept block-free; this header has none that matter).
    text = re.sub(r"//[^\n]*", "", text)

    # kPrimitiveTypeFeatures = {{TYPE, "name"}, ...}  → ordered list of names.
    prim_block = re.search(r"kPrimitiveTypeFeatures\s*=\s*\{(.*?)\};", text, re.S)
    primitive_names = re.findall(r'"([^"]+)"', prim_block.group(1)) if prim_block else []

    # Extract each FEATURIZE_FUNCTION_PAIR[_2ARG|_PTR](Name, types..., { BODY })
    # by locating the macro head and brace-matching the trailing { ... } body.
    bodies: Dict[str, str] = {}
    for m in re.finditer(r"FEATURIZE_FUNCTION_PAIR(?:_2ARG|_PTR)?\s*\(", text):
        head = text[m.end():]
        # The featurizer name is the first identifier argument.
        name_m = re.match(r"\s*([A-Za-z_]\w*)", head)
        if not name_m:
            continue
        fname = name_m.group(1)
        # Find the body brace block: first '{' after the head, brace-matched.
        brace_start = head.find("{")
        if brace_start < 0:
            continue
        depth, i = 0, brace_start
        while i < len(head):
            if head[i] == "{":
                depth += 1
            elif head[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies[fname] = head[brace_start + 1:i]
    return primitive_names, bodies


# ─────────────────────────────────────────────────────────────────────────────
# 3. Emulate NamesFor<root> for a given include_features mask
# ─────────────────────────────────────────────────────────────────────────────
# Fixed-width name helpers: template-base -> function(N, prefix) -> [names].
def _w_aggregate(n: int, prefix: str) -> List[str]:
    return [f"{prefix}_{i}" for i in range(n)] + [f"{prefix}_sum", f"{prefix}_product"]


def _w_noaggregate(n: int, prefix: str) -> List[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _w_bools(n: int, prefix: str) -> List[str]:
    return [f"{prefix}_{i}" for i in range(n)] + [
        f"{prefix}_true_count",
        f"{prefix}_false_count",
    ]


WIDTH_HELPERS = {
    "FixedNumericSequenceFeatures": _w_aggregate,
    "FixedNumericSequenceFeaturesNoAggregate": _w_noaggregate,
    "FeaturizeVariableLengthBools": _w_bools,
}


def _combine(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}_{name}"


def _iter_units(body: str):
    """Yield ('if', FLAG, subbody) and ('stmt', text) in source order.

    Handles `if (include_features & FLAG) { ... }` blocks via brace matching and
    splits the remainder into paren-balanced, semicolon-terminated statements.
    """
    i, n = 0, len(body)
    buf = []
    while i < n:
        # Detect an `if (include_features & FLAG) {` block.
        m = re.match(
            r"\s*if\s*\(\s*include_features\s*&\s*(FEATURE_\w+)\s*\)\s*\{",
            body[i:],
        )
        if m:
            if "".join(buf).strip():
                yield ("stmt", "".join(buf))
                buf = []
            flag = m.group(1)
            j = i + m.end()  # just past the '{'
            depth = 1
            while j < n and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            yield ("if", flag, body[i + m.end(): j - 1])
            i = j
            continue
        ch = body[i]
        buf.append(ch)
        if ch == ";":
            # statement boundary at top level (paren depth handled implicitly:
            # the macro calls here never contain a bare ';' inside parens)
            yield ("stmt", "".join(buf))
            buf = []
        i += 1
    if "".join(buf).strip():
        yield ("stmt", "".join(buf))


def _first_string(s: str) -> str | None:
    m = re.search(r'"([^"]*)"', s)
    return m.group(1) if m else None


def expand(
    fname: str,
    prefix: str,
    mask: int,
    bodies: Dict[str, str],
    primitive_names: List[str],
    bits: Dict[str, int],
) -> List[str]:
    """Recursively emit the ordered feature names for featurizer `fname`."""
    # Special case: FeaturizePrimitiveType is a loop over kPrimitiveTypeFeatures.
    if fname == "FeaturizePrimitiveType":
        return [_combine(prefix, nm) for nm in primitive_names]

    if fname not in bodies:
        raise KeyError(f"featurizer body not found: {fname}")

    names: List[str] = []
    for unit in _iter_units(bodies[fname]):
        if unit[0] == "if":
            _, flag, sub = unit
            if mask & bits.get(flag, 0):
                names.extend(
                    _expand_block(sub, prefix, mask, bodies, primitive_names, bits)
                )
        else:
            names.extend(
                _expand_stmt(unit[1], prefix, mask, bodies, primitive_names, bits)
            )
    return names


def _expand_block(block, prefix, mask, bodies, primitive_names, bits) -> List[str]:
    out: List[str] = []
    for unit in _iter_units(block):
        if unit[0] == "if":
            _, flag, sub = unit
            if mask & bits.get(flag, 0):
                out.extend(_expand_block(sub, prefix, mask, bodies, primitive_names, bits))
        else:
            out.extend(_expand_stmt(unit[1], prefix, mask, bodies, primitive_names, bits))
    return out


def _expand_stmt(stmt, prefix, mask, bodies, primitive_names, bits) -> List[str]:
    s = stmt.strip()
    if not s:
        return []

    # Scalar feature:  out->push_back(FEATURIZE_SWITCH("name", expr));
    if "push_back" in s and "FEATURIZE_SWITCH" in s:
        name = _first_string(s)
        return [_combine(prefix, name)] if name is not None else []

    # Dispatch:  FEATURIZE_DISPATCH(...) / FEATURIZE_DISPATCH_COND(...)
    md = re.search(r"FEATURIZE_DISPATCH(_COND)?\s*\(", s)
    if md:
        args_head = s[md.end():]
        first_arg = re.match(r"\s*([A-Za-z_]\w*)(<\s*(\d+)\s*>)?", args_head)
        callee = first_arg.group(1)
        width = first_arg.group(3)
        p = _first_string(s) or ""
        new_prefix = _combine(prefix, p)
        base = re.sub(r"<.*$", "", callee)
        if base in WIDTH_HELPERS:
            return WIDTH_HELPERS[base](int(width), new_prefix)
        # Otherwise it's a sub-featurizer → recurse with the extended prefix.
        return expand(callee, new_prefix, mask, bodies, primitive_names, bits)

    return []  # variable decls, control scaffolding, etc.


# ─────────────────────────────────────────────────────────────────────────────
# 4. Convolution-feature classification (Phase 8b input surface)
# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ CRITICAL SEMANTIC NOTE (empirically established, Phase 8a):
#   The `convolution_dim_numbers_*` columns store DIMENSION-NUMBER AXIS INDICES
#   (which tensor axis is batch/feature/spatial — small ints 0..3), NOT the
#   channel counts or spatial extents. The actual SIZES Phase 8b needs are:
#     • kernel spatial extent (filter H, W)  -> window_size_0 / window_size_1
#     • output tensor extents [N, H, W, C]   -> shape_dimensions_0..5
#     • stride / padding / dilation          -> window_stride/padding_*/dilation
#   The conv-axis-number columns are the KEY for interpreting which
#   shape_dimensions entry is the channel vs spatial axis.
#   Verified on ResNet conv1: window_size=7x7, output_shape=[16,112,112,64].
CONV_GROUPS = {
    "kernel_spatial_size": ["window_size_"],       # filter H, W  (the MXU tile rows)
    "tensor_shape": ["shape_dimensions_"],         # output [N,H,W,C] extents
    "window_stride": ["window_stride_"],
    "window_padding": ["window_padding_low_", "window_padding_high_"],
    "window_dilation": ["window_window_dilation_", "window_base_dilation_"],
    "conv_axis_numbers": ["convolution_dim_numbers"],  # axis indices (interpreters)
    "groups": ["feature_group_count", "batch_group_count"],  # depthwise / grouped
}

# Columns Phase 8b consumes to build MXU-alignment features, with their role.
# (sizes, not axis indices — see the semantic note above.)
MXU_INPUTS = {
    "kernel_h": "window_size_0",
    "kernel_w": "window_size_1",
    "out_dim_0": "shape_dimensions_0",
    "out_dim_1": "shape_dimensions_1",
    "out_dim_2": "shape_dimensions_2",
    "out_dim_3": "shape_dimensions_3",
    "stride_h": "window_stride_0",
    "stride_w": "window_stride_1",
    "out_feature_axis": "convolution_dim_numbers_output_feature_dim",  # → which out_dim is C_out
    "feature_group_count": "feature_group_count",                      # depthwise factor
}


def classify_conv(schema: List[str]) -> dict:
    groups = {g: [] for g in CONV_GROUPS}
    for idx, name in enumerate(schema):
        for g, subs in CONV_GROUPS.items():
            if any(sub in name for sub in subs):
                groups[g].append({"index": idx, "name": name})
    mxu = {}
    for role, key in MXU_INPUTS.items():
        if key in schema:
            mxu[role] = {"index": schema.index(key), "name": key}
    return {"groups": groups, "mxu_alignment_inputs": mxu}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optional empirical validation against a real ResNet npz
# ─────────────────────────────────────────────────────────────────────────────
def validate_against_npz(npz_path: str, schema: List[str], conv: dict) -> None:
    import numpy as np

    print(f"\n── Empirical validation against {os.path.basename(npz_path)} ──")
    npz = np.load(npz_path, mmap_mode="r")
    opcode = np.asarray(npz["node_opcode"])
    feat = npz["node_feat"]
    conv_nodes = np.where(opcode == CONV_OPCODE)[0]
    if len(conv_nodes) == 0:
        print(f"  no convolution nodes (opcode=={CONV_OPCODE}) in this graph.")
        return
    print(f"  {len(conv_nodes)} convolution nodes found.")

    kh = schema.index("window_size_0")
    kw = schema.index("window_size_1")
    shp = [schema.index(f"shape_dimensions_{i}") for i in range(4)]

    print(f"  kernel size cols=({kh},{kw}); output shape cols={shp}:")
    for nid in conv_nodes[:8]:
        row = np.asarray(feat[nid]).astype(int)
        print(f"    node {int(nid):>6}: kernel={row[kh]}x{row[kw]}  "
              f"output_shape={[int(row[c]) for c in shp]}")
    # Sanity: kernel spatial extents should be small positive integers (1,3,5,7,11).
    kvals = np.asarray(feat[conv_nodes, kh]).astype(int)
    ok = np.isin(kvals, [1, 2, 3, 4, 5, 7, 11]).mean()
    print(f"  window_size_0 ∈ common conv kernel extents for "
          f"{ok*100:.0f}% of conv nodes "
          f"({'PLAUSIBLE — index alignment confirmed' if ok > 0.5 else 'SUSPECT — re-check mapping'}).")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="module_layout_cost",
                    help="featurizer task selecting the include_features mask")
    ap.add_argument("--validate", nargs="?", const="auto", default=None,
                    help="cross-check conv columns against a ResNet npz "
                         "(omit path to auto-discover one)")
    args = ap.parse_args()

    if not FEATURIZERS_H.exists():
        print(f"ERROR: {FEATURIZERS_H} not found (run from repo root).", file=sys.stderr)
        return 1

    opc = parse_hlo_opcode(HLO_OPCODE_H)
    bits, counts, task_mask = opc["bits"], opc["counts"], opc["task_mask"]
    if args.task not in task_mask:
        print(f"ERROR: unknown task {args.task!r}; known: {list(task_mask)}", file=sys.stderr)
        return 1
    mask = task_mask[args.task]
    active = [f for f, b in bits.items() if mask & b]
    print(f"Task {args.task!r} → include_features mask = {mask} ({'|'.join(active) or 'NONE'})")

    primitive_names, bodies = parse_featurizers(FEATURIZERS_H)
    print(f"Parsed {len(bodies)} featurizer bodies; "
          f"{len(primitive_names)} primitive-type features.")

    schema = expand(ROOT_FEATURIZER, "", mask, bodies, primitive_names, bits)

    # ── Self-check against the reference's declared counts (the rigor guard) ──
    n_min = counts.get("kMinimalNodeFeatureCount")
    n_opnz = counts.get("kOpLevelNonZeroNodeFeatureCount")
    expected = n_min + n_opnz
    if len(schema) != expected:
        print(f"\nFATAL: reconstructed {len(schema)} features, reference declares "
              f"{n_min}+{n_opnz}={expected}. Parser drifted — NOT trustworthy.",
              file=sys.stderr)
        return 2
    print(f"✓ self-check: reconstructed {len(schema)} features "
          f"= kMinimal({n_min}) + kOpLevelNonZero({n_opnz}). Mapping is sound.")

    conv = classify_conv(schema)
    n_conv = sum(len(v) for v in conv["groups"].values())
    print(f"\nConvolution-relevant columns: {n_conv}")
    for g, items in conv["groups"].items():
        if items:
            lo, hi = items[0]["index"], items[-1]["index"]
            print(f"  {g:10s}: {len(items):2d} cols  (idx {lo}–{hi})")
    print("\nMXU-alignment input columns (Phase 8b consumes these):")
    for role, info in conv["mxu_alignment_inputs"].items():
        print(f"  [{info['index']:>3}] {role:20s} <- {info['name']}")

    OUT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCHEMA.write_text(json.dumps(
        {"task": args.task, "mask": mask, "n_features": len(schema),
         "index_to_name": {i: nm for i, nm in enumerate(schema)}}, indent=2))
    OUT_CONV.write_text(json.dumps(
        {"task": args.task, "conv_groups": conv["groups"],
         "mxu_alignment_inputs": conv["mxu_alignment_inputs"]}, indent=2))
    print(f"\nWrote {OUT_SCHEMA} and {OUT_CONV}.")

    if args.validate is not None:
        path = args.validate
        if path == "auto":
            import glob
            cands = glob.glob("data/predict-ai-model-runtime/npz_all/npz/layout/"
                              "xla/*/*/inference_mlperf_resnet*.npz")
            path = cands[0] if cands else None
        if path:
            validate_against_npz(path, schema, conv)
        else:
            print("\n(no ResNet npz found for --validate)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
