#!/usr/bin/env bash
# run_phase6b.sh — Phase 6b: DISAGGREGATE the xla:default effect.
# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 interim found the flat-vs-hardware null BREAKS on xla:default
# (Δτ=+0.165, p<0.001, all 5 seeds — see experiments/phase6_interim_xla_default.md).
# But the "hardware" arm bundles THREE differences vs flat: DiffPool hierarchy,
# GINE edge conditioning, and hardware contention edges. This script runs the
# missing arm — PLAIN DiffPool (model_phase2, 235,553 params, exact Phase 2c
# regime) — on xla/default, same 5 seeds, so the win can be attributed:
#     hierarchical ≈ hardware (~0.20)  → hierarchy explains it
#     hierarchical ≈ flat     (~0.03)  → the edge machinery explains it
#     intermediate                     → both contribute
#
# 5 jobs, sequential, each in a 24 GB capped scope via launch_capped.sh.
# .done markers → re-invoke to resume. Writes a 3-arm summary at the end, then
# STOPS (does NOT resume the nlp_default sweep — Phase 8c is queued next).
#
# Usage:  nohup bash run_phase6b.sh > logs/capped/phase6b_master.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SEEDS=(42 100 123 456 789)
COLL="xla/default"
MODEL="hierarchical"          # plain DiffPool, no edge features — the Phase 2c arm
CKPT_BASE="checkpoints/expansion/xla_default"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "════════ Phase 6b disaggregation — plain DiffPool on ${COLL}, seeds ${SEEDS[*]} ════════"
done_n=0; fail_n=0
for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="${CKPT_BASE}/${MODEL}-seed${SEED}"
    if [[ -f "${CKPT_DIR}/.done" ]]; then log "SKIP seed ${SEED} (.done)"; continue; fi
    log "START ${MODEL}-seed${SEED}"
    t0=$(date +%s)
    FOREGROUND=1 bash launch_capped.sh "${COLL}" "${MODEL}" "${SEED}"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [[ ${rc} -eq 0 && -f "${CKPT_DIR}/best_model.pt" ]]; then
        date > "${CKPT_DIR}/.done"; done_n=$((done_n+1)); log "DONE seed ${SEED} (${dt}s)"
    else
        fail_n=$((fail_n+1)); log "FAIL seed ${SEED} (${dt}s, rc=${rc}) — re-invoke to retry"
    fi
done

log "Phase 6b runs finished: ${done_n} done, ${fail_n} failed. Building 3-arm summary…"
python3 - <<'PYEOF' | tee "${CKPT_BASE}/phase6b_summary.txt"
import torch, statistics
from pathlib import Path
try:
    from scipy import stats as st
except Exception:
    st = None
base = Path("checkpoints/expansion/xla_default")
seeds = [42, 100, 123, 456, 789]
def tau(m, s):
    p = base / f"{m}-seed{s}" / "best_model.pt"
    try: return float(torch.load(p, map_location="cpu")["val_tau"])
    except Exception: return None
arms = {"flat (wide-baseline)": "wide-baseline",
        "plain DiffPool (6b)":  "hierarchical",
        "hardware DiffPool":    "hardware"}
print("Phase 6b — 3-arm disaggregation on xla:default (5 matched seeds)\n")
vals = {}
for label, m in arms.items():
    v = [tau(m, s) for s in seeds]
    vals[label] = v
    ok = [x for x in v if x is not None]
    if ok:
        print(f"  {label:<24} τ = {statistics.mean(ok):.4f} ± "
              f"{statistics.stdev(ok):.4f}   ({[f'{x:+.3f}' if x is not None else '—' for x in v]})")
hier, flat, hw = vals["plain DiffPool (6b)"], vals["flat (wide-baseline)"], vals["hardware DiffPool"]
if st and all(x is not None for x in hier):
    for other_label, other in [("flat", flat), ("hardware", hw)]:
        pairs = [(h, o) for h, o in zip(hier, other) if o is not None]
        H = [p[0] for p in pairs]; O = [p[1] for p in pairs]
        d = statistics.mean([h - o for h, o in pairs])
        t, p = st.ttest_rel(H, O)
        print(f"\n  plain-DiffPool vs {other_label}: Δτ = {d:+.4f}, paired t p={p:.4f}")
    print("\n  READ: hier≈hardware → HIERARCHY explains the xla:default win;")
    print("        hier≈flat     → EDGE MACHINERY (GINE + contention edges) explains it.")
PYEOF
log "Phase 6b COMPLETE. GPU is now FREE — next queued: Phase 8c (see experiments/phase8c_preregistration.md)."
log "nlp_default sweep is intentionally NOT resumed; resume later with:"
log "  setsid nohup bash run_expansion_capped.sh > logs/capped/sweep_master.log 2>&1 < /dev/null &"
log "  setsid nohup bash scripts/sweep_watchdog.sh > logs/capped/watchdog.out 2>&1 < /dev/null &"
