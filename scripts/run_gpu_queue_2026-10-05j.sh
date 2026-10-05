#!/usr/bin/env bash
# GPU queue, 2026-10-05 (j): waits for queue (i) (Phase 6f hardware LR grid) to exit, then
#   1. the 6f boundary step, only if the frozen rule fires (6f §4; applied by scripts/lr_star_boundary.py),
#   2. BACKBURNER B1: the amended nlp:default sweep (6d §10.5, Amendment 2 5144bc1), all 13 runs,
#      registered cutoff 2026-10-23 23:59 ET (no partial-seed result may be reported).
# The 6f boundary step goes first so the Level-1 result is not delayed by B1's ~3 days.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe (runs with a .done marker are skipped). Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-05j.sh >> logs/gpu_queue_2026-10-05j.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PAUSE_FILE=PAUSE_GPU_QUEUE
wait_if_paused() {
    if [[ -f "${PAUSE_FILE}" ]]; then
        log "PAUSED (${PAUSE_FILE} exists); waiting before: $*"
        while [[ -f "${PAUSE_FILE}" ]]; do sleep 60; done
        log "RESUMED"
    fi
}

# train COLL MODEL SEED LR RUN_TAG CKPT_BASE_DIR EXTRA
train() {
    local coll=$1 model=$2 seed=$3 lr=$4 tag=$5 base=$6 extra=$7
    local csafe=${coll//\//_}
    local dir="${base}/${csafe}/${tag}-seed${seed}"
    [[ -f "${dir}/.done" ]] && { log "SKIP ${dir}"; return 0; }
    wait_if_paused "${dir}"
    log "START ${dir} (lr ${lr})"
    EXTRA_ARGS="--lr ${lr} ${extra}" RUN_TAG="${tag}" CKPT_BASE_DIR="${base}" \
        FOREGROUND=1 bash launch_capped.sh "${coll}" "${model}" "${seed}"
    local rc=$?
    if ! python scripts/check_run_lr.py "logs/capped/${csafe}_${tag}_seed${seed}.log" "${lr}"; then
        log "ABORT: LR ${lr} not confirmed for ${dir}"; exit 1
    fi
    if [[ ${rc} -eq 0 && -f "${dir}/best_model.pt" ]]; then date > "${dir}/.done"; log "DONE ${dir}"
    else log "FAIL ${dir} rc=${rc}"; fi
}

while pgrep -f run_gpu_queue_2026-10-05i.sh >/dev/null; do sleep 120; done
log "queue (i) finished"

log "6f §4 boundary rule"
if BOUNDARY_LR=$(python scripts/lr_star_boundary.py \
        --log-pattern 'logs/capped/xla_default_hardware-lr{lr}_seed{seed}.log' \
        --base-log-pattern 'logs/capped/xla_default_hardware_seed{seed}.log'); then
    if [[ -n "${BOUNDARY_LR}" ]]; then
        log "6f boundary step fires: hardware lr ${BOUNDARY_LR}"
        for SEED in 42 100 123 456 789; do
            train xla/default hardware "${SEED}" "${BOUNDARY_LR}" "hardware-lr${BOUNDARY_LR}" checkpoints/hwlr ""
        done
    else
        log "6f boundary step not triggered"
    fi
else
    log "6f grid incomplete; boundary rule NOT applied (evaluate manually before B1 results are used)"
fi

log "B1 / 6d §10.5: amended nlp:default sweep (cutoff 2026-10-23 23:59 ET)"
# Stage dir maps <base>/nlp_default onto the registered checkpoints/expansion/nlp_default_lr1e-4/.
mkdir -p checkpoints/expansion/nlp_default_lr1e-4 checkpoints/_stage_lr1e-4
[[ -L checkpoints/_stage_lr1e-4/nlp_default ]] || ln -s ../expansion/nlp_default_lr1e-4 checkpoints/_stage_lr1e-4/nlp_default
for SEED in 42 100 123 456 789; do
    train nlp/default wide-baseline "${SEED}" 1e-4 wide-baseline-lr1e-4 checkpoints/_stage_lr1e-4 --fast-eval
    train nlp/default hardware      "${SEED}" 1e-4 hardware-lr1e-4      checkpoints/_stage_lr1e-4 --fast-eval
done
for SEED in 123 456 789; do
    train nlp/default wide-baseline "${SEED}" 1e-3 wide-baseline checkpoints/expansion --fast-eval
done
log "queue finished. Next: evaluate 6f (if not yet), then 6d §10.5 against its frozen rules"
