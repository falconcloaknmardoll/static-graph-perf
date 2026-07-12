#!/usr/bin/env bash
# run_expansion_capped.sh — Phase 6 expansion sweep, memory-SAFE edition.
# ─────────────────────────────────────────────────────────────────────────────
# Runs the full 5-seed paired flat-vs-hardware ablation across the 3 expansion
# collections, but routes EVERY job through launch_capped.sh's 24 GB systemd
# cgroup scope (FOREGROUND=1 = one job at a time). This is the multi-day
# replacement for run_collection_expansion.sh, which crashed the desktop via
# host-RAM OOM. See project_phase6_crash_fix memory / launch_capped.sh.
#
# Robustness for an unattended multi-day run:
#   • each completed run drops a `.done` marker → re-invoking SKIPS it (resume).
#   • the whole loop is meant to be launched under nohup (see the echo at the end
#     OR just: nohup bash run_expansion_capped.sh > logs/capped/sweep_master.log 2>&1 &).
#   • per-job logs live in logs/capped/; this driver writes a master progress log.
#
# Usage
#   nohup bash run_expansion_capped.sh > logs/capped/sweep_master.log 2>&1 &
#   bash run_expansion_capped.sh xla/default          # subset
#   DRY_RUN=1 bash run_expansion_capped.sh            # list what WOULD run
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail   # NOT -e: one failed run must not abort the remaining sweep

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SEEDS=(42 100 123 456 789)
MODELS=(wide-baseline hardware)
if [[ $# -gt 0 ]]; then COLLECTIONS=("$@"); else COLLECTIONS=("xla/default" "nlp/random" "nlp/default"); fi

CKPT_BASE="checkpoints/expansion"
LOG_DIR="logs/capped"
mkdir -p "${LOG_DIR}"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

total=$(( ${#COLLECTIONS[@]} * ${#SEEDS[@]} * ${#MODELS[@]} ))
done_n=0; skip_n=0; fail_n=0; idx=0
log "════════ Phase 6 capped sweep — ${total} runs (${#COLLECTIONS[@]} coll × ${#SEEDS[@]} seeds × ${#MODELS[@]} models) ════════"
log "memory cap per job: MemoryMax=${MEM_MAX:-24G} (each job in its own systemd scope)"

for COLL in "${COLLECTIONS[@]}"; do
    COLL_SAFE="${COLL//\//_}"
    if [[ ! -d "data/predict-ai-model-runtime/npz_all/npz/layout/${COLL}/train" ]]; then
        log "[SKIP-COLLECTION] ${COLL} — no train data"; continue
    fi
    log "──────── collection ${COLL} ────────"
    for SEED in "${SEEDS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            idx=$((idx + 1))
            CKPT_DIR="${CKPT_BASE}/${COLL_SAFE}/${MODEL}-seed${SEED}"
            DONE_MARK="${CKPT_DIR}/.done"
            tag="[${idx}/${total}] ${COLL_SAFE}/${MODEL}-seed${SEED}"

            if [[ -f "${DONE_MARK}" ]]; then
                log "SKIP  ${tag} (.done present)"; skip_n=$((skip_n + 1)); continue
            fi
            if [[ "${DRY_RUN:-0}" == "1" ]]; then
                log "WOULD RUN  ${tag}"; continue
            fi

            log "START ${tag}"
            t0=$(date +%s)
            FOREGROUND=1 bash launch_capped.sh "${COLL}" "${MODEL}" "${SEED}"
            rc=$?
            dt=$(( $(date +%s) - t0 ))
            if [[ ${rc} -eq 0 && -f "${CKPT_DIR}/best_model.pt" ]]; then
                date > "${DONE_MARK}"
                done_n=$((done_n + 1))
                log "DONE  ${tag}  (${dt}s, rc=0)"
            else
                fail_n=$((fail_n + 1))
                log "FAIL  ${tag}  (${dt}s, rc=${rc}, no .done written — will retry on re-invoke)"
            fi
        done
    done
done

log "════════ sweep finished: ${done_n} done, ${skip_n} skipped, ${fail_n} failed (of ${total}) ════════"
log "per-collection paired summaries: re-use the python block in run_collection_expansion.sh,"
log "or inspect checkpoints/expansion/<coll>/<model>-seed<seed>/best_model.pt val_tau."
[[ ${fail_n} -gt 0 ]] && log "NOTE: ${fail_n} run(s) failed — just re-invoke this script to retry only those."
exit 0
