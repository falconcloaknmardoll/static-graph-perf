#!/usr/bin/env bash
# run_ablation_seeds.sh
# ─────────────────────────────────────────────────────────────────────────────
# Ablation study: parameter-matched flat vs. DiffPool hierarchical
#
# Runs both models for 5 seeds to produce mean ± std for Kendall's τ,
# establishing whether the hierarchy (not parameter count) drives the gain.
#
# Models
#   wide-baseline : BaselineGraphSAGE_Large (h=172, ~236k params, flat GNN)
#   hierarchical  : HierarchicalGraphSAGE   (h=128, ~236k params, DiffPool)
#
# Dataset fidelity: no --max-configs-per-graph cap — the model sees the full
# configuration search space every epoch, including BERT's ~18k configs.
# --configs-per-batch 8 prevents OOM on BERT-scale graphs.
# --num-workers 4 parallelises data loading to reduce GPU idle time.
#
# WandB
#   Project : tpugraphs-ablation
#   Run names: wide-baseline-seed{SEED}  /  diffpool-seed{SEED}
#   Disable with: export WANDB_MODE=disabled
#
# Output
#   Checkpoints : checkpoints/ablation/{wide-baseline,diffpool}-seed{SEED}/best_model.pt
#   Summary     : checkpoints/ablation/results_summary.txt  (τ table at the end)
#
# Estimated runtime: ~23 hrs total (10 runs × ~140 min/run on RTX 5080)
#
# Usage
#   bash run_ablation_seeds.sh                  # full run
#   DRY_RUN=1 bash run_ablation_seeds.sh        # print commands only
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
WANDB_PROJECT="tpugraphs-ablation"
CKPT_BASE="checkpoints/ablation"
SUMMARY_FILE="${CKPT_BASE}/results_summary.txt"

# Python binary — use the project virtualenv if it exists
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${CKPT_BASE}"

{
    echo "Ablation results — $(date)"
    echo "Seeds: ${SEEDS[*]}"
    echo "Epochs: ${EPOCHS}   configs/batch: ${CONFIGS_PER_BATCH}   num-workers: ${NUM_WORKERS}"
    echo "max-configs/graph: ${MAX_CONFIGS_PER_GRAPH} (shuffled each epoch — different 128 configs drawn per epoch)"
    echo ""
    printf "%-28s  %6s  %s\n" "run" "seed" "best_val_tau"
    printf "%-28s  %6s  %s\n" "---" "----" "------------"
} > "${SUMMARY_FILE}"

log "Starting ablation — ${#SEEDS[@]} seeds × 2 models = $((${#SEEDS[@]} * 2)) runs"
log "Estimated wall time: ~19 hrs (cap=128 configs/graph, shuffled each epoch)"

# ── Main loop ─────────────────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do

    # ── 1. Wide baseline (flat, parameter-matched) ────────────────────────────
    WIDE_CKPT_DIR="${CKPT_BASE}/wide-baseline-seed${SEED}"
    WIDE_RUN_NAME="wide-baseline-seed${SEED}"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Seed ${SEED}  |  wide-baseline"
    mkdir -p "${WIDE_CKPT_DIR}"

    CMD_WIDE="${PYTHON} train.py \
        --model wide-baseline \
        --epochs ${EPOCHS} \
        --configs-per-batch ${CONFIGS_PER_BATCH} \
        --max-configs-per-graph ${MAX_CONFIGS_PER_GRAPH} \
        --num-workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --grad-clip 0.1 \
        --eval-every 5 \
        --ckpt-dir ${WIDE_CKPT_DIR} \
        --project ${WANDB_PROJECT} \
        --run-name ${WIDE_RUN_NAME}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: ${CMD_WIDE}"
    else
        eval "${CMD_WIDE}" 2>&1 | tee "${WIDE_CKPT_DIR}/run.log"
    fi

    # Extract best val_tau from the checkpoint
    WIDE_TAU=$(${PYTHON} -c "
import torch
try:
    ckpt = torch.load('${WIDE_CKPT_DIR}/best_model.pt', map_location='cpu')
    print(f\"{ckpt.get('val_tau', float('nan')):.4f}\")
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || grep -o 'best=[+-][0-9.]*' "${WIDE_CKPT_DIR}/run.log" | tail -1 | cut -d= -f2 || echo "N/A")

    log "  → wide-baseline seed=${SEED}  τ=${WIDE_TAU}"
    printf "%-28s  %6s  %s\n" "${WIDE_RUN_NAME}" "${SEED}" "${WIDE_TAU}" >> "${SUMMARY_FILE}"

    # ── 2. Hierarchical (DiffPool) ────────────────────────────────────────────
    HIER_CKPT_DIR="${CKPT_BASE}/diffpool-seed${SEED}"
    HIER_RUN_NAME="diffpool-seed${SEED}"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Seed ${SEED}  |  hierarchical (DiffPool)"
    mkdir -p "${HIER_CKPT_DIR}"

    CMD_HIER="${PYTHON} train.py \
        --model hierarchical \
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
        --ckpt-dir ${HIER_CKPT_DIR} \
        --project ${WANDB_PROJECT} \
        --run-name ${HIER_RUN_NAME}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: ${CMD_HIER}"
    else
        eval "${CMD_HIER}" 2>&1 | tee "${HIER_CKPT_DIR}/run.log"
    fi

    HIER_TAU=$(${PYTHON} -c "
import torch
try:
    ckpt = torch.load('${HIER_CKPT_DIR}/best_model.pt', map_location='cpu')
    print(f\"{ckpt.get('val_tau', float('nan')):.4f}\")
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || grep -o 'best=[+-][0-9.]*' "${HIER_CKPT_DIR}/run.log" | tail -1 | cut -d= -f2 || echo "N/A")

    log "  → diffpool     seed=${SEED}  τ=${HIER_TAU}"
    printf "%-28s  %6s  %s\n" "${HIER_RUN_NAME}" "${SEED}" "${HIER_TAU}" >> "${SUMMARY_FILE}"

done

# ── Final summary ─────────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All runs complete.  Computing mean ± std …"

${PYTHON} - <<'PYEOF' 2>/dev/null | tee -a "${SUMMARY_FILE}"
import statistics
from pathlib import Path
import torch

base = Path("checkpoints/ablation")
results = {"wide-baseline": [], "diffpool": []}

for ckpt_dir in sorted(base.iterdir()):
    if not ckpt_dir.is_dir():
        continue
    ckpt = ckpt_dir / "best_model.pt"
    if not ckpt.exists():
        continue
    d = torch.load(ckpt, map_location="cpu")
    tau = d.get("val_tau")
    if tau is None:
        continue
    if "wide-baseline" in ckpt_dir.name:
        results["wide-baseline"].append(float(tau))
    elif "diffpool" in ckpt_dir.name:
        results["diffpool"].append(float(tau))

print("\n" + "─" * 52)
print("  Ablation summary (C2 verification)")
print("─" * 52)
for model, taus in results.items():
    if not taus:
        print(f"  {model:<22}  no completed runs")
        continue
    mn = statistics.mean(taus)
    sd = statistics.stdev(taus) if len(taus) > 1 else float("nan")
    print(f"  {model:<22}  n={len(taus)}  τ = {mn:.4f} ± {sd:.4f}")
    print(f"               individual: {[f'{t:.4f}' for t in taus]}")

if results["wide-baseline"] and results["diffpool"]:
    wide_mn = statistics.mean(results["wide-baseline"])
    hier_mn = statistics.mean(results["diffpool"])
    delta = hier_mn - wide_mn
    print(f"\n  Δτ (diffpool − wide-baseline) = {delta:+.4f}")
    if delta > 0:
        print("  ↳ Hierarchy helps (sign positive); check if gap > noise.")
    else:
        print("  ↳ Hierarchy DOES NOT help (sign negative or zero).")
print("─" * 52)
PYEOF

log "Summary written to ${SUMMARY_FILE}"
log "Done."
