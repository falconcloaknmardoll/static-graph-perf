#!/usr/bin/env bash
# GPU queue, 2026-10-05 (i): Phase 6f, hardware arm LR grid on xla:default (prereg 9b11a96).
# Waits for queue (h) (6e boundary step) to exit, then runs hardware at lr {1e-4, 3e-4, 3e-3} x 5 seeds.
# A boundary step (3e-5 or 1e-2), if 6f §4 triggers it, is added by a later queue after evaluation.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe (runs with a .done marker are skipped). Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-05i.sh >> logs/gpu_queue_2026-10-05i.log 2>&1 &
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

while pgrep -f run_gpu_queue_2026-10-05h.sh >/dev/null; do sleep 120; done
log "queue (h) finished; Phase 6f: hardware LR grid"
for LR in 1e-4 3e-4 3e-3; do
    for SEED in 42 100 123 456 789; do
        train xla/default hardware "${SEED}" "${LR}" "hardware-lr${LR}" checkpoints/hwlr ""
    done
done

log "queue finished. Next: evaluate 6f against experiments/phase6f_hardware_lr_preregistration.md (LR*_hw, boundary rule, convergence check)"
