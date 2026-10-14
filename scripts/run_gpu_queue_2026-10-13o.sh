#!/usr/bin/env bash
# GPU queue, 2026-10-13 (o): O19, 8c/8d CONV Top-1 re-scored at eval batch 8
# (experiments/o19_8c8d_batch8_declaration.md). Eval-only, existing checkpoints.
# Waits for queues k, l, m and n so it never shares the GPU with training or with R0, and so it
# cannot displace 8e, R4, R3 or R0. Not started after 2026-10-23 23:59 ET.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Re-invoking is safe: finished caches in checkpoints/o19_eval_cache/ are reused. Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-13o.sh >> logs/gpu_queue_2026-10-13o.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUTOFF=$(TZ=America/New_York date -d "2026-10-23 23:59" +%s)
while pgrep -f "^bash scripts/run_gpu_queue_2026-10-(11[klm]|13n).sh" >/dev/null; do sleep 120; done
log "queues (k, l, m, n) finished"
# Belt and braces: no training or R0 process may still hold the GPU.
while pgrep -f "^[^ ]*python[^ ]* (train(_8c)?\.py|scripts/r0_conv_top1_eval\.py)" >/dev/null; do sleep 120; done
if [[ -f PAUSE_GPU_QUEUE ]]; then
    log "PAUSED (PAUSE_GPU_QUEUE exists); waiting before O19"
    while [[ -f PAUSE_GPU_QUEUE ]]; do sleep 60; done
    log "RESUMED"
fi
if (( $(date +%s) > CUTOFF )); then log "CUTOFF passed; O19 not started"; exit 0; fi
[[ -f experiments/o19_8c8d_batch8_results.json ]] && { log "O19 results already exist; nothing to do"; exit 0; }
log "START O19 (GPU, batch-8 re-score of 8c/8d)"
systemd-run --user --scope -q -p MemoryMax=24G -p MemoryHigh=22G \
    python scripts/o19_batch8_rescore.py > logs/o19_batch8_rescore.log 2>&1
rc=$?
if [[ ${rc} -eq 0 && -f experiments/o19_8c8d_batch8_results.json ]]; then
    log "DONE O19 (log logs/o19_batch8_rescore.log)"
else
    log "FAIL O19 rc=${rc} (see logs/o19_batch8_rescore.log)"
fi
log "queue finished. Next: check the reproduction gate, then apply the declaration §4 table"
