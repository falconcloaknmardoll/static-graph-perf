#!/usr/bin/env bash
# GPU queue, 2026-09-29 (g): the "musts" only, after the 2026-09-29 reboot killed queue (f) mid-run.
# Phase 6e flat-LR runs (24fa8a4) -> O9 Top-K nlp:random. Everything else from queue (f) is deferred;
# see experiments/BACKBURNER.md for what was dropped and how to relaunch it.
#
# PAUSE SWITCH (for lab-mates who need the GPU): `touch PAUSE_GPU_QUEUE` in the repo root. The queue
# finishes the run in progress, then waits before starting the next one. `rm PAUSE_GPU_QUEUE` resumes.
# To free the GPU immediately instead, stop the current run; it will restart from scratch on resume.
#
# Detached with setsid/nohup so it survives the Claude session. Re-invoking is safe: runs with a .done
# marker are skipped. Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-09-29g.sh >> logs/gpu_queue_2026-09-29g.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# The machine is shared, so cap CPU threads. Not a scientific variable: numerics are unchanged.
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
    # The LR override must be confirmed in the log before anything else proceeds.
    if ! python scripts/check_run_lr.py "logs/capped/${csafe}_${tag}_seed${seed}.log" "${lr}"; then
        log "ABORT: LR ${lr} not confirmed for ${dir}"; exit 1
    fi
    if [[ ${rc} -eq 0 && -f "${dir}/best_model.pt" ]]; then date > "${dir}/.done"; log "DONE ${dir}"
    else log "FAIL ${dir} rc=${rc}"; fi
}

log "Phase 6e: flat-LR runs"
for LR in 1e-4 3e-4 3e-3; do
    for SEED in 42 100 123 456 789; do
        train xla/default wide-baseline "${SEED}" "${LR}" "wide-baseline-lr${LR}" checkpoints/flatlr ""
    done
done

# Declared O9 cell (9fc0075).
if [[ ! -f experiments/topk_slowdown_nlp_random.json ]]; then
    wait_if_paused "O9 top-k nlp_random"
    log "O9: top-k nlp_random"; python scripts/topk_slowdown.py --coll nlp_random
fi

log "queue finished (musts). Next: evaluate 6e against its frozen pre-registration; the boundary step, if triggered, is in experiments/BACKBURNER.md"
