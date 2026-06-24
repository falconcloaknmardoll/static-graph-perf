#!/usr/bin/env bash
# run_phase3_seeds.sh
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 definitive experiment — does hardware-aware edge augmentation beat the
# parameter-matched flat baseline?
#
# Runs ONLY the new model (HardwareAwareDiffPool, ~234k params) for 5 seeds.
# The flat baseline was already run in Phase 2c (wide-baseline, τ = 0.2914 ± 0.0343,
# same seeds, same regime) — we reuse those checkpoints for a PAIRED comparison.
#
# Regime (identical to Phase 2c so the comparison is controlled):
#   --epochs 50 --configs-per-batch 8 --max-configs-per-graph 128 --num-workers 4
#   --lp-weight 0.1 --entropy-weight 0.01
#
# WandB     : project tpugraphs-phase3-ablation, runs hardware-seed{SEED}
# Output    : checkpoints/phase3/hardware-seed{SEED}/best_model.pt
#             checkpoints/phase3/results_summary.txt
#
# Usage
#   bash run_phase3_seeds.sh                 # full run
#   DRY_RUN=1 bash run_phase3_seeds.sh       # print commands only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS=(42 100 123 456 789)
EPOCHS=50
CONFIGS_PER_BATCH=8
MAX_CONFIGS_PER_GRAPH=128
NUM_WORKERS=4
LP_WEIGHT=0.1
ENTROPY_WEIGHT=0.01
WANDB_PROJECT="tpugraphs-phase3-ablation"
CKPT_BASE="checkpoints/phase3"
WIDE_BASE="checkpoints/ablation"          # Phase 2c flat-baseline checkpoints (for pairing)
WIDE_MEAN=0.2914                          # Phase 2c wide-baseline mean τ
SUMMARY_FILE="${CKPT_BASE}/results_summary.txt"

if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${CKPT_BASE}"

{
    echo "Phase 3 results — $(date)"
    echo "Model: HardwareAwareDiffPool (~234k params)"
    echo "Seeds: ${SEEDS[*]}"
    echo "Epochs: ${EPOCHS}  configs/batch: ${CONFIGS_PER_BATCH}  cap: ${MAX_CONFIGS_PER_GRAPH}  workers: ${NUM_WORKERS}"
    echo "Compared against Phase 2c wide-baseline mean τ = ${WIDE_MEAN}"
    echo ""
    printf "%-22s  %6s  %s\n" "run" "seed" "best_val_tau"
    printf "%-22s  %6s  %s\n" "---" "----" "------------"
} > "${SUMMARY_FILE}"

log "Starting Phase 3 — ${#SEEDS[@]} hardware runs"
log "Estimated wall time: ~10-12 hrs (GINEConv is heavier than SAGEConv; ~2 hr/run)"

# ── Main loop ─────────────────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    HW_CKPT_DIR="${CKPT_BASE}/hardware-seed${SEED}"
    HW_RUN_NAME="hardware-seed${SEED}"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Seed ${SEED}  |  hardware (Phase 3)"
    mkdir -p "${HW_CKPT_DIR}"

    CMD="${PYTHON} train.py \
        --model hardware \
        --epochs ${EPOCHS} \
        --configs-per-batch ${CONFIGS_PER_BATCH} \
        --max-configs-per-graph ${MAX_CONFIGS_PER_GRAPH} \
        --num-workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --grad-clip 0.1 \
        --lp-weight ${LP_WEIGHT} \
        --entropy-weight ${ENTROPY_WEIGHT} \
        --eval-every 5 \
        --ckpt-dir ${HW_CKPT_DIR} \
        --project ${WANDB_PROJECT} \
        --run-name ${HW_RUN_NAME}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: ${CMD}"
    else
        eval "${CMD}" 2>&1 | tee "${HW_CKPT_DIR}/run.log"
    fi

    HW_TAU=$(${PYTHON} -c "
import torch
try:
    ckpt = torch.load('${HW_CKPT_DIR}/best_model.pt', map_location='cpu')
    print(f\"{ckpt.get('val_tau', float('nan')):.4f}\")
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || grep -o 'best=[+-][0-9.]*' "${HW_CKPT_DIR}/run.log" | tail -1 | cut -d= -f2 || echo "N/A")

    log "  → hardware seed=${SEED}  τ=${HW_TAU}"
    printf "%-22s  %6s  %s\n" "${HW_RUN_NAME}" "${SEED}" "${HW_TAU}" >> "${SUMMARY_FILE}"
done

# ── Final summary ─────────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All Phase 3 runs complete.  Computing mean ± std and Δτ …"

WIDE_MEAN="${WIDE_MEAN}" ${PYTHON} - <<'PYEOF' 2>/dev/null | tee -a "${SUMMARY_FILE}"
import os, statistics
from pathlib import Path
import torch

SEEDS = [42, 100, 123, 456, 789]
hw_base   = Path("checkpoints/phase3")
wide_base = Path("checkpoints/ablation")
wide_mean = float(os.environ.get("WIDE_MEAN", "0.2914"))

def tau(p):
    try:
        return float(torch.load(p, map_location="cpu")["val_tau"])
    except Exception:
        return None

hw   = {s: tau(hw_base   / f"hardware-seed{s}"      / "best_model.pt") for s in SEEDS}
wide = {s: tau(wide_base / f"wide-baseline-seed{s}" / "best_model.pt") for s in SEEDS}

hw_vals = [v for v in hw.values() if v is not None]

print("\n" + "─" * 56)
print("  Phase 3 summary — hardware-aware DiffPool")
print("─" * 56)
if not hw_vals:
    print("  no completed hardware runs")
    raise SystemExit

hw_mn = statistics.mean(hw_vals)
hw_sd = statistics.stdev(hw_vals) if len(hw_vals) > 1 else float("nan")
print(f"  hardware (Phase 3)   n={len(hw_vals)}  τ = {hw_mn:.4f} ± {hw_sd:.4f}")
print(f"               individual: {[f'{v:.4f}' for v in hw_vals]}")

# ── Δτ vs the known wide-baseline group mean (as requested) ──
print(f"\n  wide-baseline mean (Phase 2c) = {wide_mean:.4f}")
print(f"  Δτ (hardware − wide-baseline mean) = {hw_mn - wide_mean:+.4f}")

# ── Bonus: PAIRED comparison reusing Phase 2c per-seed checkpoints ──
paired = [(s, hw[s], wide[s]) for s in SEEDS if hw[s] is not None and wide[s] is not None]
if len(paired) >= 2:
    diffs = [h - w for _, h, w in paired]
    print(f"\n  Paired vs Phase 2c wide-baseline (same seeds, n={len(paired)}):")
    for s, h, w in paired:
        tag = "hardware" if h > w else "flat"
        print(f"    seed {s:>3}:  hw {h:.4f}  vs  flat {w:.4f}   Δ {h-w:+.4f}  ({tag} wins)")
    dmean = statistics.mean(diffs)
    dsd   = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
    print(f"    mean paired Δτ = {dmean:+.4f} ± {dsd:.4f}")
    try:
        from scipy import stats
        t, p = stats.ttest_rel([h for _, h, _ in paired], [w for _, _, w in paired])
        w_, pw = stats.wilcoxon([h for _, h, _ in paired], [w for _, _, w in paired])
        print(f"    paired t-test: t={t:.3f} p={p:.3f}   |   Wilcoxon: W={w_:.1f} p={pw:.3f}")
        verdict = ("Hardware edges HELP (significant)" if (dmean > 0 and p < 0.05)
                   else "Hardware edges help (trend, not significant)" if dmean > 0
                   else "Hardware edges do NOT help")
        print(f"    ↳ {verdict}")
    except Exception as e:
        print(f"    (scipy unavailable for significance test: {e})")
print("─" * 56)
PYEOF

log "Summary written to ${SUMMARY_FILE}"
log "Done."
