#!/usr/bin/env bash
# launch_capped.sh — run ONE train.py job inside a hard memory-capped systemd scope.
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Phase 6 on the large nlp collections saturated Host RAM; with only ~2 GB
# swap the kernel OOM-killer culled DESKTOP apps to survive ("everything got
# laggy, apps closed themselves"). This wrapper sandboxes the training process in
# a transient cgroup v2 scope with a hard MemoryMax. If the job spikes past the
# ceiling, the cgroup OOM-killer kills a process INSIDE THE SCOPE — never your
# editor/browser. The desktop becomes un-crashable by this job.
#
# It also hardcodes a safe DataLoader worker count for the nlp collections (the
# fork-per-worker RAM multiplication was the trigger) and routes everything
# through nohup-friendly logging.
#
# USAGE
#   bash launch_capped.sh <collection> <model> <seed> [epochs]
#     <collection>  nlp/random | nlp/default | xla/random | xla/default
#     <model>       wide-baseline | hardware
#     <seed>        e.g. 42
#     [epochs]      optional (default 50)
#
# ENV OVERRIDES
#   MEM_MAX=24G            hard cgroup ceiling (OOM-kills the scope if breached)
#   MEM_HIGH=22G           soft throttle: kernel reclaims hard before the kill
#   MEM_SWAP_MAX=8G        cap swap the scope may use (needs the 32G swapfile live)
#   NUM_WORKERS=<n>        force worker count (else: 1 for nlp, 4 for xla)
#   CONFIGS_PER_BATCH=8  MAX_CONFIGS_PER_GRAPH=128  EVAL_BATCH=8  EVAL_EVERY=5
#   SMOKE=1               smoke mode: --no-wandb, --epochs defaults to 3,
#                         --eval-every 1 (hit the BERT eval stressor immediately)
#   DRY_RUN=1             print the systemd-run command and exit
#
# OUTPUT
#   logs/capped/<coll>_<model>_seed<seed>[_smoke].log     (tee'd, nohup-safe)
#   checkpoints/expansion/<coll>/<model>-seed<seed>/best_model.pt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COLL="${1:?collection required, e.g. nlp/random}"
MODEL="${2:?model required: wide-baseline | hardware}"
SEED="${3:?seed required, e.g. 42}"
EPOCHS_ARG="${4:-}"

# ── Memory ceiling ────────────────────────────────────────────────────────────
MEM_MAX="${MEM_MAX:-24G}"
MEM_HIGH="${MEM_HIGH:-22G}"
MEM_SWAP_MAX="${MEM_SWAP_MAX:-8G}"

# ── Worker default: 0 (NO worker processes) ───────────────────────────────────
# Hard lesson from the 2026-06-29 sweep: when a job is OOM-killed, its DataLoader
# worker children are re-parented and SURVIVE, holding RAM and keeping the scope
# "active". Across runs these orphans accumulated (~10 GiB) until the host OOM'd
# and SIGKILL'd everything (rc=137). num-workers=0 spawns NO worker processes, so
# there is nothing to orphan — the strongest guarantee. Data loads in the main
# process; for these small graphs the GPU forward dominates anyway.
NUM_WORKERS="${NUM_WORKERS:-0}"

CONFIGS_PER_BATCH="${CONFIGS_PER_BATCH:-8}"
MAX_CONFIGS_PER_GRAPH="${MAX_CONFIGS_PER_GRAPH:-128}"
# Phase-8c hooks (defaults preserve original behavior exactly):
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"     # e.g. train_8c.py
EXTRA_ARGS="${EXTRA_ARGS:-}"                 # e.g. --mxu-arm full
CKPT_BASE_DIR="${CKPT_BASE_DIR:-checkpoints/expansion}"
RUN_TAG="${RUN_TAG:-${MODEL}}"               # names ckpt dir / unit / run distinctly per arm
EVAL_BATCH="${EVAL_BATCH:-8}"        # small: BERT-scale eval batches are the GPU/host stressor
LP_WEIGHT="${LP_WEIGHT:-0.1}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.01}"
WANDB_PROJECT="${WANDB_PROJECT:-tpugraphs-expansion}"
DATA_BASE="data/predict-ai-model-runtime/npz_all/npz/layout"

# ── Smoke vs full ─────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
    EPOCHS="${EPOCHS_ARG:-3}"
    EVAL_EVERY="${EVAL_EVERY:-1}"     # force the validation (BERT) memory event at epoch 1
    WANDB_FLAG="--no-wandb"
    LOG_SUFFIX="_smoke"
else
    EPOCHS="${EPOCHS_ARG:-50}"
    EVAL_EVERY="${EVAL_EVERY:-5}"
    WANDB_FLAG=""
    LOG_SUFFIX=""
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLL_SAFE="${COLL//\//_}"
DATA_ROOT="${DATA_BASE}/${COLL}"
CKPT_DIR="${CKPT_BASE_DIR}/${COLL_SAFE}/${RUN_TAG}-seed${SEED}"
LOG_DIR="logs/capped"
LOG_FILE="${LOG_DIR}/${COLL_SAFE}_${RUN_TAG}_seed${SEED}${LOG_SUFFIX}.log"
RUN_NAME="${COLL_SAFE}-${RUN_TAG}-seed${SEED}${LOG_SUFFIX}"
UNIT="tpu-${COLL_SAFE}-${RUN_TAG}-s${SEED}${LOG_SUFFIX}"

mkdir -p "${ROOT}/${CKPT_DIR}" "${ROOT}/${LOG_DIR}"

if [[ ! -d "${ROOT}/${DATA_ROOT}/train" ]]; then
    echo "ERROR: no data at ${DATA_ROOT}/train" >&2; exit 1
fi

PYTHON="python"
[[ -x "${ROOT}/.venv/bin/python" ]] && PYTHON="${ROOT}/.venv/bin/python"

# DiffPool-family models (hardware, hierarchical) carry the two aux-loss
# weights; flat baselines ignore them. (train.py defaults are the same values —
# explicit here so run logs show the regime.)
EXTRA=""
[[ "${MODEL}" == "hardware" || "${MODEL}" == "hierarchical" ]] && \
    EXTRA="--lp-weight ${LP_WEIGHT} --entropy-weight ${ENTROPY_WEIGHT}"

# Inner command — cd guarantees cwd inside the transient scope; exec so signals
# (and the cgroup OOM-killer) hit python directly, not a wrapper shell.
INNER="cd '${ROOT}' && exec ${PYTHON} ${TRAIN_SCRIPT} \
    --model ${MODEL} \
    --data-root ${DATA_ROOT} \
    --epochs ${EPOCHS} \
    --configs-per-batch ${CONFIGS_PER_BATCH} \
    --max-configs-per-graph ${MAX_CONFIGS_PER_GRAPH} \
    --eval-batch ${EVAL_BATCH} \
    --num-workers ${NUM_WORKERS} \
    --eval-every ${EVAL_EVERY} \
    --seed ${SEED} \
    --dropout 0.0 --lr 1e-3 --weight-decay 1e-4 --grad-clip 0.1 \
    ${EXTRA} \
    --ckpt-dir ${CKPT_DIR} \
    --project ${WANDB_PROJECT} --run-name ${RUN_NAME} ${WANDB_FLAG} ${EXTRA_ARGS}"

# systemd-run: transient cgroup scope with a hard memory ceiling.
#   MemoryMax  → hard limit; breach ⇒ cgroup OOM-kill INSIDE the scope only.
#   MemoryHigh → soft limit; kernel throttles+reclaims before the hard kill.
#   MemorySwapMax → bound swap so we can't reintroduce swap-thrash on the desktop.
SYSTEMD_CMD=(systemd-run --user --scope
    --unit="${UNIT}"
    -p MemoryMax="${MEM_MAX}"
    -p MemoryHigh="${MEM_HIGH}"
    -p MemorySwapMax="${MEM_SWAP_MAX}"
    -p MemoryAccounting=yes
    bash -c "${INNER}")

echo "─────────────────────────────────────────────────────────────"
echo " collection : ${COLL}   model: ${MODEL}   seed: ${SEED}"
echo " epochs     : ${EPOCHS}   eval-every: ${EVAL_EVERY}   workers: ${NUM_WORKERS}"
echo " mem cap    : MemoryMax=${MEM_MAX}  High=${MEM_HIGH}  SwapMax=${MEM_SWAP_MAX}"
echo " scope unit : ${UNIT}.scope"
echo " log        : ${LOG_FILE}"
echo " monitor    : systemctl --user show ${UNIT}.scope -p MemoryCurrent"
echo "─────────────────────────────────────────────────────────────"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${SYSTEMD_CMD[@]}"; echo; exit 0
fi

cd "${ROOT}"
if [[ "${FOREGROUND:-0}" == "1" ]]; then
    # Synchronous: run the scope, block until it finishes, tee to the log.
    # Used by run_expansion_capped.sh so the sweep runs ONE job at a time
    # (the driver loop itself is what gets nohup'd).
    "${SYSTEMD_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    rc="${PIPESTATUS[0]}"
    # GUARANTEE teardown: force-stop the scope so NO process (orphaned worker,
    # zombie) survives into the next job. Safe no-op if already gone.
    systemctl --user stop "${UNIT}.scope" 2>/dev/null || true
    systemctl --user reset-failed "${UNIT}.scope" 2>/dev/null || true
    exit "${rc}"
else
    # nohup so a terminal / VS Code session cycle (what killed the ep17 run)
    # cannot SIGHUP the job. Backgrounds the whole thing.
    nohup "${SYSTEMD_CMD[@]}" > "${LOG_FILE}" 2>&1 &
    WRAP_PID=$!
    echo "launched (wrapper pid ${WRAP_PID}) → tail -f ${LOG_FILE}"
fi
