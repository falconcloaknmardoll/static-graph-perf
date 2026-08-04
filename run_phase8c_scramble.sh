#!/usr/bin/env bash
# run_phase8c_scramble.sh — Phase 8c specificity control (prereg amendment A2).
# ─────────────────────────────────────────────────────────────────────────────
# ONE new arm — `scramble` (k=4, dim 162): the `channel` columns with per-graph
# rows permuted among MXU-resolved nodes (value distribution + placement +
# config-invariance preserved; node↔value alignment destroyed). Tests P4:
# channel > scramble ⇒ real MXU co-design; channel ≈ scramble ⇒ capacity artifact.
#
# Reuses the EXACT capped harness as run_phase8c.sh (num-workers 0, same seeds,
# same collection, same ckpt layout) so the new runs are paired with the frozen
# baseline/channel checkpoints. Only 5 runs (baseline/channel/spatial/full exist).
#
# Usage:  setsid nohup bash run_phase8c_scramble.sh > logs/capped/phase8c_scramble_master.log 2>&1 < /dev/null &
# Then:   python phase8c_evaluate.py --arms baseline channel scramble
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SEEDS=(42 100 123 456 789)
ARM="scramble"
COLL="xla/random"
CKPT_BASE="checkpoints/phase8c"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

mkdir -p "${CKPT_BASE}"
log "════════ Phase 8c control — scramble arm, 5 runs on ${COLL} (prereg A2) ════════"
done_n=0; fail_n=0; idx=0
for SEED in "${SEEDS[@]}"; do
    idx=$((idx+1))
    TAG="mxu-${ARM}"
    CKPT_DIR="${CKPT_BASE}/xla_random/${TAG}-seed${SEED}"
    if [[ -f "${CKPT_DIR}/.done" ]]; then log "SKIP [${idx}/5] ${TAG}-seed${SEED}"; continue; fi
    log "START [${idx}/5] ${TAG}-seed${SEED}"
    t0=$(date +%s)
    FOREGROUND=1 \
    TRAIN_SCRIPT="train_8c.py" \
    EXTRA_ARGS="--mxu-arm ${ARM}" \
    RUN_TAG="${TAG}" \
    CKPT_BASE_DIR="${CKPT_BASE}" \
    WANDB_PROJECT="tpugraphs-phase8c" \
        bash launch_capped.sh "${COLL}" "wide-baseline" "${SEED}"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [[ ${rc} -eq 0 && -f "${CKPT_DIR}/best_model.pt" ]]; then
        date > "${CKPT_DIR}/.done"; done_n=$((done_n+1)); log "DONE  [${idx}/5] ${TAG}-seed${SEED} (${dt}s)"
    else
        fail_n=$((fail_n+1)); log "FAIL  [${idx}/5] ${TAG}-seed${SEED} (${dt}s rc=${rc}) — re-invoke to retry"
    fi
done
log "════════ scramble control complete: ${done_n} done, ${fail_n} failed ════════"
log "NEXT: python phase8c_evaluate.py --arms baseline channel scramble --out experiments/phase8c_control_results.md"
log "      (CONV-subset Top-1: P4 = channel significantly < scramble ⇒ real MXU info)"
