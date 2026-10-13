#!/usr/bin/env bash
# GPU queue, 2026-10-13 (n): R0 remaining cells on GPU (r0_conv_top1_structure_declaration.md, Amendment 3).
# Waits for queues k, l and m so R0 has the GPU to itself (the Oct 11 OOM came from sharing it with 8e).
# Eval-only. Cached cells (flat + hierarchical at batch 32, checkpoints/r0_eval_cache/) are reused;
# the script computes hardware at batch 32 and all arms at batch 8, then writes
# experiments/r0_conv_top1_results.{md,json}.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes.
# Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-13n.sh >> logs/gpu_queue_2026-10-13n.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

while pgrep -f "^bash scripts/run_gpu_queue_2026-10-11[klm].sh" >/dev/null; do sleep 120; done
log "queues (k, l, m) finished"
# Belt and braces: no training process may still hold the GPU.
while pgrep -f "^[^ ]*python[^ ]* train(_8c)?\.py" >/dev/null; do sleep 120; done

if [[ -f PAUSE_GPU_QUEUE ]]; then
    log "PAUSED (PAUSE_GPU_QUEUE exists); waiting before R0"
    while [[ -f PAUSE_GPU_QUEUE ]]; do sleep 60; done
    log "RESUMED"
fi

[[ -f experiments/r0_conv_top1_results.json ]] && { log "R0 results already exist; nothing to do"; exit 0; }

log "START R0 (GPU, remaining cells)"
systemd-run --user --scope -q -p MemoryMax=24G -p MemoryHigh=22G \
    python scripts/r0_conv_top1_eval.py > logs/r0_conv_top1_eval_gpu.log 2>&1
rc=$?
if [[ ${rc} -eq 0 && -f experiments/r0_conv_top1_results.json ]]; then
    log "DONE R0 (log logs/r0_conv_top1_eval_gpu.log)"
else
    log "FAIL R0 rc=${rc} (see logs/r0_conv_top1_eval_gpu.log)"
fi
log "queue finished. Next: evaluate R0 against its declaration §4 (+ Amendments 1–3)"
