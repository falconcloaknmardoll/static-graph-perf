#!/usr/bin/env bash
# GPU queue, 2026-10-11 (k): Phase 8e, LR robustness of the 8c channel effect (prereg 44267c7).
# baseline + channel (8c pipeline, train_8c.py, xla:random) at lr {1e-4, 3e-4} x 5 seeds = 20 runs.
# lr 1e-3 cells reuse checkpoints/phase8c (prereg §4). Arms interleaved within each seed (§4).
# A boundary step (3e-5 or 3e-3 per arm), if §4 triggers it, is added by a later queue after LR*.
# Cutoff (§8.5): every registered run must finish by 2026-10-23 23:59 ET.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe (runs with a .done marker are skipped). Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-11k.sh >> logs/gpu_queue_2026-10-11k.log 2>&1 &
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

# train8e SEED LR ARM
train8e() {
    local seed=$1 lr=$2 arm=$3
    local tag="mxu-${arm}-lr${lr}" base=checkpoints/phase8e
    local dir="${base}/xla_random/${tag}-seed${seed}"
    [[ -f "${dir}/.done" ]] && { log "SKIP ${dir}"; return 0; }
    wait_if_paused "${dir}"
    log "START ${dir} (lr ${lr})"
    TRAIN_SCRIPT="train_8c.py" EXTRA_ARGS="--lr ${lr} --mxu-arm ${arm}" RUN_TAG="${tag}" \
        CKPT_BASE_DIR="${base}" WANDB_PROJECT="tpugraphs-phase8e" \
        FOREGROUND=1 bash launch_capped.sh xla/random wide-baseline "${seed}"
    local rc=$?
    if ! python scripts/check_run_lr.py "logs/capped/xla_random_${tag}_seed${seed}.log" "${lr}"; then
        log "ABORT: LR ${lr} not confirmed for ${dir}"; exit 1
    fi
    if [[ ${rc} -eq 0 && -f "${dir}/best_model.pt" ]]; then date > "${dir}/.done"; log "DONE ${dir}"
    else log "FAIL ${dir} rc=${rc}"; fi
}

# Do not share the GPU with the O9 tuned Top-K re-check if it is still running.
# Match only the python process (a shell wrapper whose command line contains the string would
# match itself forever).
while pgrep -f "^[^ ]*python[^ ]* (-u )?scripts/topk_slowdown.py" >/dev/null; do sleep 60; done
log "Phase 8e: baseline + channel LR grid on xla:random"
for SEED in 42 100 123 456 789; do
    for LR in 1e-4 3e-4; do
        for ARM in baseline channel; do
            train8e "${SEED}" "${LR}" "${ARM}"
        done
    done
done

log "queue finished. Next: pick LR*_arm per 8e §4 (overall val tau), apply the boundary rule, then evaluate Q1/Q2"
