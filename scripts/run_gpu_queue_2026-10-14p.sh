#!/usr/bin/env bash
# GPU queue, 2026-10-14 (p): O21, Phase 6b per-graph tau at eval batch 8 (experiments/o21_6b_batch8_declaration.md).
# Eval-only. Waits for queues k, l, m, n, o and the 8e evaluation window. Not started after 2026-10-23 23:59 ET.
#
# PAUSE SWITCH: `touch PAUSE_GPU_QUEUE` in the repo root; `rm PAUSE_GPU_QUEUE` resumes. Launch:
#   setsid nohup bash scripts/run_gpu_queue_2026-10-14p.sh >> logs/gpu_queue_2026-10-14p.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUTOFF=$(TZ=America/New_York date -d "2026-10-23 23:59" +%s)
while pgrep -f "^bash scripts/(run_gpu_queue_2026-10-(11[klm]|13[no])|run_8e_eval_window)\.sh" >/dev/null; do sleep 120; done
log "queues (k-o) and the 8e window finished"
while pgrep -f "^[^ ]*python[^ ]* (train(_8c)?\.py|scripts/(r0_conv_top1_eval|o19_batch8_rescore|phase8e_evaluate)\.py)" >/dev/null; do sleep 120; done
if [[ -f PAUSE_GPU_QUEUE ]]; then
    log "PAUSED (PAUSE_GPU_QUEUE exists); waiting before O21"
    while [[ -f PAUSE_GPU_QUEUE ]]; do sleep 60; done
    log "RESUMED"
fi
if (( $(date +%s) > CUTOFF )); then log "CUTOFF passed; O21 not started"; exit 0; fi
[[ -f experiments/phase6b_confound_checks_b8.json ]] && { log "O21 results already exist; nothing to do"; exit 0; }
log "START O21 (6b per-graph tau, batch 8)"
systemd-run --user --scope -q -p MemoryMax=24G -p MemoryHigh=22G \
    python scripts/phase6b_batch8_rescore.py > logs/phase6b_batch8_rescore.log 2>&1
rc=$?
if [[ ${rc} -eq 0 && -f experiments/phase6b_confound_checks_b8.json ]]; then
    log "DONE O21 (log logs/phase6b_batch8_rescore.log)"
else
    log "FAIL O21 rc=${rc} (see logs/phase6b_batch8_rescore.log)"
fi
log "queue finished. Next: check the gate, then apply the declaration §2"
