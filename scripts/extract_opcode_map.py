#!/usr/bin/env python3
"""
scripts/extract_opcode_map.py
-----------------------------
Phase 3 — extract the *exact* opcode string→integer mapping that Google used
when generating the TPUGraphs dataset.

WHY THIS EXISTS
~~~~~~~~~~~~~~~
The integers stored in `node_opcode` do NOT follow the standard alphabetical
XLA HLO enum. TPUGraphs uses a custom index assignment defined in
`reference/tpu_graphs/process_data/xla/hlo_opcode.h` via two X-macro lists:

    #define LEARNED_HLO_OPCODE_LIST(V)   \
        V("abs", 1)                       \
        ...
        V("convolution", 26)              \
        V("dot", 34)                      \
        ...

    #define TOMBSTONE_HLO_OPCODE_LIST(V) \
        V("batch-norm-grad", ...)         \
        ...

`StringToOpcodeID()` builds a hash map from exactly these `V(name, id)` pairs.
This script parses both lists and emits the name→id dict — the single source
of truth for `MXU_OPS` / `MEM_OPS` in dataset.py.

USAGE
~~~~~
    python scripts/extract_opcode_map.py             # pretty-print the map
    python scripts/extract_opcode_map.py --check      # assert convolution/dot ids
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict

# Repo-root-relative path to the ground-truth header.
HLO_OPCODE_H = (
    Path(__file__).resolve().parent.parent
    / "reference" / "tpu_graphs" / "process_data" / "xla" / "hlo_opcode.h"
)

# Matches a single X-macro entry:  V("convolution", 26)
#   - tolerant of arbitrary whitespace and the trailing line-continuation "\"
_ENTRY_RE = re.compile(r'V\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)')


def extract_opcode_map(header_path: Path = HLO_OPCODE_H) -> Dict[str, int]:
    """Parse every V("name", id) entry from hlo_opcode.h into a name→id dict.

    Both LEARNED_HLO_OPCODE_LIST and TOMBSTONE_HLO_OPCODE_LIST contribute
    entries; StringToOpcodeID() merges them, so we do too.
    """
    if not header_path.is_file():
        sys.exit(f"ERROR: opcode header not found at {header_path}")

    text = header_path.read_text()
    mapping: Dict[str, int] = {}
    for name, idx in _ENTRY_RE.findall(text):
        idx = int(idx)
        if name in mapping and mapping[name] != idx:
            sys.exit(
                f"ERROR: conflicting ids for {name!r}: "
                f"{mapping[name]} vs {idx}"
            )
        mapping[name] = idx

    if not mapping:
        sys.exit("ERROR: no V(\"name\", id) entries parsed — regex/file mismatch.")
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="assert known anchor opcodes (convolution=26, dot=34)")
    args = ap.parse_args()

    m = extract_opcode_map()
    inv = {v: k for k, v in m.items()}

    print(f"Parsed {len(m)} opcodes from {HLO_OPCODE_H.name}")
    print(f"id range: [{min(m.values())}, {max(m.values())}]\n")
    for idx in sorted(inv):
        print(f"  {idx:3d}  {inv[idx]}")

    if args.check:
        # Anchors the parser against known values from the header.
        assert m.get("convolution") == 26, m.get("convolution")
        assert m.get("dot") == 34, m.get("dot")
        print("\n[check] convolution=26, dot=34 — OK")


if __name__ == "__main__":
    main()
