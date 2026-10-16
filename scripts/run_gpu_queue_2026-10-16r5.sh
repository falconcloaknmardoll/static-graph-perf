#!/usr/bin/env bash
# GPU queue, 2026-10-16 (r5): R5, tuned oracle vs tuned baseline (experiments/r5_oracle_tuned_declaration.md,
# frozen 58cb565). lr 3e-5, 15 new runs. No run starts after 2026-10-23 23:59 ET.
# Order (§8): oracle on 8e's 5 seeds, then baseline+oracle pairs on the 5 new seeds.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe (runs with a .done marker are skipped). Launch (detached, survives the session):
#   systemd-run --user --unit=gpu-queue-r5 --collect bash scripts/run_gpu_queue_2026-10-16r5.sh
#   (log: logs/gpu_queue_2026-10-16r5.log)
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> logs/gpu_queue_2026-10-16r5.log 2>&1
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUTOFF=$(TZ=America/New_York date -d "2026-10-23 23:59" +%s)
LR=3e-5; BASE=checkpoints/r5

PAUSE_FILE=PAUSE_GPU_QUEUE
wait_if_paused() {
    if [[ -f "${PAUSE_FILE}" ]]; then
        log "PAUSED (${PAUSE_FILE} exists); waiting before: $*"
        while [[ -f "${PAUSE_FILE}" ]]; do sleep 60; done
        log "RESUMED"
    fi
}

run() {  # run <arm> <seed>; one retry from scratch on failure (declaration §7.2)
    local arm=$1 seed=$2 tag script args
    if [[ ${arm} == oracle ]]; then tag="oracle-bucket-lr${LR}"; script=train_8d.py; args="--lr ${LR}"
    else tag="mxu-baseline-lr${LR}"; script=train_8c.py; args="--lr ${LR} --mxu-arm baseline"; fi
    local dir="${BASE}/xla_random/${tag}-seed${seed}" logf="logs/capped/xla_random_${tag}_seed${seed}.log"
    [[ -f "${dir}/.done" ]] && { log "SKIP ${dir}"; return; }
    for attempt in 1 2; do
        if (( $(date +%s) > CUTOFF )); then log "CUTOFF passed; not starting ${dir}"; return; fi
        wait_if_paused "${dir}"
        log "START ${dir} (lr ${LR}, attempt ${attempt})"
        [[ ${attempt} == 2 ]] && rm -rf "${dir}"
        TRAIN_SCRIPT="${script}" EXTRA_ARGS="${args}" RUN_TAG="${tag}" CKPT_BASE_DIR="${BASE}" \
            WANDB_PROJECT="tpugraphs-r5" FOREGROUND=1 bash launch_capped.sh xla/random wide-baseline "${seed}"
        local rc=$?
        if ! python scripts/check_run_lr.py "${logf}" "${LR}"; then log "ABORT: LR not confirmed for ${dir}"; exit 1; fi
        if [[ ${rc} -eq 0 && -f "${dir}/best_model.pt" ]]; then date > "${dir}/.done"; log "DONE ${dir}"; return; fi
        log "FAIL ${dir} rc=${rc}"
    done
}

log "R5 queue start"
for SEED in 42 100 123 456 789; do run oracle "${SEED}"; done
for SEED in 7 21 314 2718 9001; do run baseline "${SEED}"; run oracle "${SEED}"; done
log "queue finished. Next: scripts/r5_evaluate.py (committed before scoring), declaration §6"
