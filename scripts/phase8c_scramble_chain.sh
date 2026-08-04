#!/usr/bin/env bash
# phase8c_scramble_chain.sh — waits for the in-flight CPU seed-42 scramble run
# (launched directly, outside run_phase8c_scramble.sh, as a GPU-vs-CPU timing
# test) to finish, marks it .done, then hands off to run_phase8c_scramble.sh
# for the remaining seeds (100/123/456/789). Lets the whole 5-seed control run
# unattended over ~2 days of CPU-only wall-clock (see PROJECT_TIMELINE.md /
# memory project-gpu-blocker for why GPU isn't available).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CKPT_DIR="checkpoints/phase8c/xla_random/mxu-scramble-seed42"
UNIT="tpu-xla_random-mxu-scramble-s42.scope"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for ${UNIT} to finish..."
while systemctl --user is-active --quiet "${UNIT}" 2>/dev/null; do
    sleep 60
done

if [[ -f "${CKPT_DIR}/best_model.pt" ]]; then
    date > "${CKPT_DIR}/.done"
    log "seed42 (scramble) complete, best_model.pt present -> marked .done"
else
    log "seed42 (scramble) ended WITHOUT best_model.pt -- NOT marking done, aborting chain"
    exit 1
fi

log "launching run_phase8c_scramble.sh for remaining seeds (100/123/456/789; 42 will be skipped)"
bash run_phase8c_scramble.sh
log "chain complete"
