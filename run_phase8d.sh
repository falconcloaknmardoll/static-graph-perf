#!/usr/bin/env bash
# run_phase8d.sh — Phase 8d: oracle rank-bucket capstone probe (PRE-REGISTERED).
# ─────────────────────────────────────────────────────────────────────────────
# Protocol is FROZEN in experiments/phase8d_preregistration.md. Only ONE new
# arm needs training: oracle-bucket (158+5 rank-bucket one-hot cols). baseline
# and channel are REUSED unchanged from checkpoints/phase8c/xla_random/ — no
# retraining, per prereg §6. 5 seeds, same regime as 8c (Amendment: none —
# prereg §7 explicitly matches 8c with no changes).
#
# ⛔ GPU down as of 2026-08-21 — runs via the CPU workaround already proven by
# 8c's confound-A run (~10hr/seed train + eval, measured 2026-08-03: CPU is
# ~6x slower than GPU, not comparable but tractable, ~3-4 days unattended for
# 5 seeds sequential).
#
# Usage:  setsid nohup bash run_phase8d.sh > logs/capped/phase8d_master.log 2>&1 < /dev/null &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SEEDS=(42 100 123 456 789)
COLL="xla/random"
CKPT_BASE="checkpoints/phase8d"
TAG="oracle-bucket"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

mkdir -p "${CKPT_BASE}"
log "════════ Phase 8d — oracle-bucket capstone, 5 seeds on ${COLL}, pre-registered ════════"
done_n=0; fail_n=0
for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="${CKPT_BASE}/xla_random/${TAG}-seed${SEED}"
    if [[ -f "${CKPT_DIR}/.done" ]]; then log "SKIP ${TAG}-seed${SEED}"; continue; fi
    log "START ${TAG}-seed${SEED}"
    t0=$(date +%s)
    FOREGROUND=1 \
    TRAIN_SCRIPT="train_8d.py" \
    RUN_TAG="${TAG}" \
    CKPT_BASE_DIR="${CKPT_BASE}" \
    WANDB_PROJECT="tpugraphs-phase8d" \
        bash launch_capped.sh "${COLL}" "wide-baseline" "${SEED}"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [[ ${rc} -eq 0 && -f "${CKPT_DIR}/best_model.pt" ]]; then
        date > "${CKPT_DIR}/.done"; done_n=$((done_n+1)); log "DONE  ${TAG}-seed${SEED} (${dt}s)"
    else
        fail_n=$((fail_n+1)); log "FAIL  ${TAG}-seed${SEED} (${dt}s rc=${rc}) — re-invoke to retry"
    fi
done
log "════════ Phase 8d training complete: ${done_n} done, ${fail_n} failed ════════"
log "NEXT: python phase8d_evaluate.py   (oracle-vs-channel Top-1 slowdown per prereg §7/§8)"
log "8d was the last item in the compute queue before GPU-blocked A7 (nlp_default) — see"
log "PROJECT_TIMELINE.md. Do NOT auto-resume nlp_default from here (7-8wk on CPU, left for GPU)."
