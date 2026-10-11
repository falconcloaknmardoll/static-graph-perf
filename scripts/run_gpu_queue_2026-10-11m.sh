#!/usr/bin/env bash
# GPU queue, 2026-10-11 (m): R3, the scramble arm re-run on GPU (experiments/r3_scramble_gpu_replication_declaration.md).
# Waits for queue l (8e boundary + R4). lr 1e-3, 5 seeds. No run starts after 2026-10-23 23:59 ET.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe (runs with a .done marker are skipped). Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-11m.sh >> logs/gpu_queue_2026-10-11m.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUTOFF=$(TZ=America/New_York date -d "2026-10-23 23:59" +%s)

PAUSE_FILE=PAUSE_GPU_QUEUE
wait_if_paused() {
    if [[ -f "${PAUSE_FILE}" ]]; then
        log "PAUSED (${PAUSE_FILE} exists); waiting before: $*"
        while [[ -f "${PAUSE_FILE}" ]]; do sleep 60; done
        log "RESUMED"
    fi
}

while pgrep -f "^bash scripts/run_gpu_queue_2026-10-11[kl].sh" >/dev/null; do sleep 120; done
log "queues (k, l) finished; R3: scramble on GPU"

TAG=mxu-scramble-gpu; BASE=checkpoints/r3
for SEED in 42 100 123 456 789; do
    DIR="${BASE}/xla_random/${TAG}-seed${SEED}"
    [[ -f "${DIR}/.done" ]] && { log "SKIP ${DIR}"; continue; }
    if (( $(date +%s) > CUTOFF )); then log "CUTOFF passed; not starting ${DIR}"; continue; fi
    wait_if_paused "${DIR}"
    log "START ${DIR} (lr 1e-3)"
    TRAIN_SCRIPT="train_8c.py" EXTRA_ARGS="--lr 1e-3 --mxu-arm scramble" RUN_TAG="${TAG}" \
        CKPT_BASE_DIR="${BASE}" WANDB_PROJECT="tpugraphs-phase8c" \
        FOREGROUND=1 bash launch_capped.sh xla/random wide-baseline "${SEED}"
    rc=$?
    if ! python scripts/check_run_lr.py "logs/capped/xla_random_${TAG}_seed${SEED}.log" 1e-3; then
        log "ABORT: LR not confirmed for ${DIR}"; exit 1
    fi
    if [[ ${rc} -eq 0 && -f "${DIR}/best_model.pt" ]]; then date > "${DIR}/.done"; log "DONE ${DIR}"
    else log "FAIL ${DIR} rc=${rc}"; fi
done
log "queue finished. Next: evaluate R3 against its declaration (R3-P4, placebo, CPU-vs-GPU offset)"
