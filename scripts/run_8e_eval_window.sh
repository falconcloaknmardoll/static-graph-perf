#!/usr/bin/env bash
# 8e evaluation in a GPU window between queue-l runs (2026-10-14).
# Sets PAUSE_GPU_QUEUE so the queues wait before their next run, waits for the running training job to
# finish, runs scripts/phase8e_evaluate.py alone on the GPU, then removes the pause (only if this script
# set it), on success or failure.
# Launch:
#   setsid nohup bash scripts/run_8e_eval_window.sh >> logs/phase8e_eval_window.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PAUSE_FILE=PAUSE_GPU_QUEUE
if [[ -f "${PAUSE_FILE}" ]]; then
    log "PAUSE already set by someone else; not touching it. Exiting."; exit 1
fi
touch "${PAUSE_FILE}"
trap 'rm -f "${PAUSE_FILE}"; log "pause removed; queues resume"' EXIT
log "pause set; waiting for the running training job to finish"
while pgrep -f "^[^ ]*python[^ ]* (train(_8c)?\.py|scripts/r0_conv_top1_eval\.py|scripts/o19_batch8_rescore\.py)" >/dev/null; do sleep 60; done
log "GPU free; START 8e evaluation"
systemd-run --user --scope -q -p MemoryMax=24G -p MemoryHigh=22G \
    python scripts/phase8e_evaluate.py > logs/phase8e_evaluate.log 2>&1
rc=$?
if [[ ${rc} -eq 0 && -f experiments/phase8e_results.json ]]; then
    log "DONE 8e evaluation (logs/phase8e_evaluate.log)"
else
    log "FAIL 8e evaluation rc=${rc} (see logs/phase8e_evaluate.log)"
fi
