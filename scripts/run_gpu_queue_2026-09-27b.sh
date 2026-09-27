#!/usr/bin/env bash
# GPU queue, 2026-09-27 (b): O1 (already running) -> O8 xla:random re-score -> O9 Top-K (xla cells)
# -> Phase 6e flat-LR runs (24fa8a4) -> 6d §10.5 amended nlp:default sweep (Amendment 2, 5144bc1)
# -> O9 Top-K nlp:random (slow, last). Detached with setsid/nohup so it survives the Claude
# session. Re-invoking is safe: runs with a .done marker are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# train COLL MODEL SEED LR RUN_TAG CKPT_BASE_DIR EXTRA
train() {
    local coll=$1 model=$2 seed=$3 lr=$4 tag=$5 base=$6 extra=$7
    local csafe=${coll//\//_}
    local dir="${base}/${csafe}/${tag}-seed${seed}"
    [[ -f "${dir}/.done" ]] && { log "SKIP ${dir}"; return 0; }
    log "START ${dir} (lr ${lr})"
    EXTRA_ARGS="--lr ${lr} ${extra}" RUN_TAG="${tag}" CKPT_BASE_DIR="${base}" \
        FOREGROUND=1 bash launch_capped.sh "${coll}" "${model}" "${seed}"
    local rc=$?
    # The LR override must be confirmed in the log before anything else proceeds.
    if ! grep -q "Fresh LR schedule: ${lr/e-/e-0}" "logs/capped/${csafe}_${tag}_seed${seed}.log" 2>/dev/null; then
        log "ABORT: LR ${lr} not confirmed for ${dir}"; exit 1
    fi
    if [[ ${rc} -eq 0 && -f "${dir}/best_model.pt" ]]; then date > "${dir}/.done"; log "DONE ${dir}"
    else log "FAIL ${dir} rc=${rc}"; fi
}

while pgrep -f "eval_batch_sensitivity.py --coll xla_default" >/dev/null; do sleep 60; done
log "O1 finished"

[[ -f experiments/eval_batch_sensitivity_xla_random.json ]] || {
    log "O8: xla_random re-score"; python scripts/eval_batch_sensitivity.py --coll xla_random --batches 8 2 32 64; }
for c in xla_default xla_random; do
    [[ -f experiments/topk_slowdown_${c}.json ]] || { log "O9: top-k ${c}"; python scripts/topk_slowdown.py --coll ${c}; }
done

log "Phase 6e: flat-LR runs"
for LR in 1e-4 3e-4 3e-3; do
    for SEED in 42 100 123 456 789; do
        train xla/default wide-baseline "${SEED}" "${LR}" "wide-baseline-lr${LR}" checkpoints/flatlr ""
    done
done

log "6d §10.5: amended nlp:default sweep (cutoff 2026-10-23 23:59 ET)"
# Stage dir maps <base>/nlp_default onto the registered checkpoints/expansion/nlp_default_lr1e-4/.
mkdir -p checkpoints/expansion/nlp_default_lr1e-4 checkpoints/_stage_lr1e-4
[[ -L checkpoints/_stage_lr1e-4/nlp_default ]] || ln -s ../expansion/nlp_default_lr1e-4 checkpoints/_stage_lr1e-4/nlp_default
for SEED in 42 100 123 456 789; do
    train nlp/default wide-baseline "${SEED}" 1e-4 wide-baseline-lr1e-4 checkpoints/_stage_lr1e-4 --fast-eval
    train nlp/default hardware      "${SEED}" 1e-4 hardware-lr1e-4      checkpoints/_stage_lr1e-4 --fast-eval
done
for SEED in 123 456 789; do
    train nlp/default wide-baseline "${SEED}" 1e-3 wide-baseline checkpoints/expansion --fast-eval
done
log "6d §10.5 sweep finished"

[[ -f experiments/topk_slowdown_nlp_random.json ]] || { log "O9: top-k nlp_random"; python scripts/topk_slowdown.py --coll nlp_random; }
log "queue finished"
