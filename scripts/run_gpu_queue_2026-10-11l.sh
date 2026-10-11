#!/usr/bin/env bash
# GPU queue, 2026-10-11 (l): runs after queue k (8e grid).
#   1. 8e boundary step(s), if 8e §4's frozen LR* rule triggers them (scripts/phase8e_lr_star.py).
#      8e has priority over everything below.
#   2. R4 (experiments/r4_heldout_conv_declaration.md + Amendment 1): baseline + channel, each at
#      its final 8e LR* (after any boundary step), 5 seeds, trained on xla/random_r4 (11 conv
#      graphs from 6 families held out). If either LR* cannot be determined, R4 does not run.
# No run is STARTED after 2026-10-23 23:59 ET (both 8e and R4 are unreportable after that).
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe (runs with a .done marker are skipped). Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-11l.sh >> logs/gpu_queue_2026-10-11l.log 2>&1 &
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

# train8c COLL SEED LR ARM TAG BASE
train8c() {
    local coll=$1 seed=$2 lr=$3 arm=$4 tag=$5 base=$6
    local csafe=${coll//\//_}
    local dir="${base}/${csafe}/${tag}-seed${seed}"
    [[ -f "${dir}/.done" ]] && { log "SKIP ${dir}"; return 0; }
    if (( $(date +%s) > CUTOFF )); then log "CUTOFF passed; not starting ${dir}"; return 0; fi
    wait_if_paused "${dir}"
    log "START ${dir} (lr ${lr})"
    TRAIN_SCRIPT="train_8c.py" EXTRA_ARGS="--lr ${lr} --mxu-arm ${arm}" RUN_TAG="${tag}" \
        CKPT_BASE_DIR="${base}" WANDB_PROJECT="tpugraphs-phase8e" \
        FOREGROUND=1 bash launch_capped.sh "${coll}" wide-baseline "${seed}"
    local rc=$?
    if ! python scripts/check_run_lr.py "logs/capped/${csafe}_${tag}_seed${seed}.log" "${lr}"; then
        log "ABORT: LR ${lr} not confirmed for ${dir}"; exit 1
    fi
    if [[ ${rc} -eq 0 && -f "${dir}/best_model.pt" ]]; then date > "${dir}/.done"; log "DONE ${dir}"
    else log "FAIL ${dir} rc=${rc}"; fi
}

while pgrep -f "^bash scripts/run_gpu_queue_2026-10-11k.sh" >/dev/null; do sleep 120; done
log "queue (k) finished; 8e LR* + boundary rule"

for ARM in baseline channel; do
    if ! B=$(python scripts/phase8e_lr_star.py --arm "${ARM}"); then
        log "8e grid incomplete for ${ARM}; boundary rule NOT applied (resolve manually)"; continue
    fi
    if [[ -n "${B}" ]]; then
        log "8e boundary step for ${ARM}: lr ${B}"
        for SEED in 42 100 123 456 789; do
            train8c xla/random "${SEED}" "${B}" "${ARM}" "mxu-${ARM}-lr${B}" checkpoints/phase8e
        done
    else
        log "8e: no boundary step for ${ARM}"
    fi
done

declare -A LRSTAR
for ARM in baseline channel; do
    if ! LRSTAR[${ARM}]=$(python scripts/phase8e_lr_star.py --arm "${ARM}" --print-lr-star); then
        log "R4 NOT RUN: 8e LR* for ${ARM} undetermined (R4 Amendment 1)"; exit 0
    fi
done
log "R4: held-out conv families at 8e LR* (baseline ${LRSTAR[baseline]}, channel ${LRSTAR[channel]})"
python scripts/make_r4_split.py
for SEED in 42 100 123 456 789; do
    for ARM in baseline channel; do
        L=${LRSTAR[${ARM}]}
        train8c xla/random_r4 "${SEED}" "${L}" "${ARM}" "mxu-${ARM}-lr${L}" checkpoints/r4
    done
done

log "queue finished. Next: evaluate 8e (Q1/Q2 at LR*) and R4 against their frozen documents"
