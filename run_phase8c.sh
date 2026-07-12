#!/usr/bin/env bash
# run_phase8c.sh — Phase 8c: MXU-feature co-design experiment (PRE-REGISTERED).
# ─────────────────────────────────────────────────────────────────────────────
# Protocol is FROZEN in experiments/phase8c_preregistration.md (+ amendment A1).
# 4 arms × 5 seeds = 20 runs of the SAME wide flat backbone, differing only in
# appended MXU input columns:
#     baseline (k=0) | channel (k=4) | spatial (k=3) | full (k=16)
# Train on layout:xla:random train (the Phase 2c/3/5 lineage that established
# the conv failure); primary metric (conv-subset Top-1 slowdown on the frozen
# split) is computed AFTERWARD by phase8c_evaluate.py — not here.
#
# Arms are interleaved WITHIN each seed so a seed's quadruple runs under
# similar box-load conditions (comparisons are paired by seed).
#
# Usage:  setsid nohup bash run_phase8c.sh > logs/capped/phase8c_master.log 2>&1 < /dev/null &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SEEDS=(42 100 123 456 789)
ARMS=(baseline channel spatial full)
COLL="xla/random"
CKPT_BASE="checkpoints/phase8c"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

mkdir -p "${CKPT_BASE}"
log "════════ Phase 8c — 20 runs (4 arms × 5 seeds) on ${COLL}, pre-registered ════════"
done_n=0; fail_n=0; idx=0
for SEED in "${SEEDS[@]}"; do
    for ARM in "${ARMS[@]}"; do
        idx=$((idx+1))
        TAG="mxu-${ARM}"
        CKPT_DIR="${CKPT_BASE}/xla_random/${TAG}-seed${SEED}"
        if [[ -f "${CKPT_DIR}/.done" ]]; then log "SKIP [${idx}/20] ${TAG}-seed${SEED}"; continue; fi
        log "START [${idx}/20] ${TAG}-seed${SEED}"
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
            date > "${CKPT_DIR}/.done"; done_n=$((done_n+1)); log "DONE  [${idx}/20] ${TAG}-seed${SEED} (${dt}s)"
        else
            fail_n=$((fail_n+1)); log "FAIL  [${idx}/20] ${TAG}-seed${SEED} (${dt}s rc=${rc}) — re-invoke to retry"
        fi
    done
done
log "════════ Phase 8c training complete: ${done_n} done, ${fail_n} failed ════════"
log "NEXT: python phase8c_evaluate.py   (conv-subset Top-1 slowdown per the prereg §7)"
log "Then resume nlp_default:"
log "  setsid nohup bash run_expansion_capped.sh > logs/capped/sweep_master.log 2>&1 < /dev/null &"
log "  setsid nohup bash scripts/sweep_watchdog.sh > logs/capped/watchdog.out 2>&1 < /dev/null &"
