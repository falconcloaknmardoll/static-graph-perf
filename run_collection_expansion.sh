#!/usr/bin/env bash
# run_collection_expansion.sh
# ─────────────────────────────────────────────────────────────────────────────
# Spine A expansion — does the noise floor / null result GENERALIZE beyond
# layout:xla:random?  Re-runs the 5-seed, parameter-matched paired ablation
# (Wide flat Baseline  vs  Hardware-aware DiffPool) on the three remaining
# layout collections.
#
#   xla:default   (7 val graphs)   ┐ same eval-set size as the original xla:random
#   nlp:random    (20 val graphs)  ├ a natural test of whether MORE eval graphs
#   nlp:default   (20 val graphs)  ┘ TIGHTEN the noise floor (key Spine-A result)
#
# Both models: ~236k params, --configs-per-batch 8, --max-configs-per-graph 128,
# --num-workers 4 (identical regime to Phase 2c/3, so cross-collection comparison
# is controlled). All four collections are 158-dim (verified), no model change.
#
# 3 collections × 5 seeds × 2 models = 30 runs.  nlp collections have ~200 train
# graphs (≈3× xla) so their epochs are ~3× longer — expect ~5-6 DAYS total.
# Each collection is independent: if a run dies, re-invoke for that one collection.
#
# Output
#   checkpoints/expansion/<coll>/<model>-seed<seed>/best_model.pt
#   checkpoints/expansion/<coll>/results_summary.txt   (per-collection paired test)
#   checkpoints/expansion/MASTER_summary.txt           (all collections, at the end)
#
# Usage
#   bash run_collection_expansion.sh                       # all 3 collections
#   bash run_collection_expansion.sh xla/default            # just one (parallel-friendly)
#   bash run_collection_expansion.sh nlp/random nlp/default # a subset
#   DRY_RUN=1 bash run_collection_expansion.sh              # print commands only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS=(42 100 123 456 789)
EPOCHS=50
CONFIGS_PER_BATCH=8
MAX_CONFIGS_PER_GRAPH=128
NUM_WORKERS=4
LP_WEIGHT=0.1
ENTROPY_WEIGHT=0.01
WANDB_PROJECT="tpugraphs-expansion"
DATA_BASE="data/predict-ai-model-runtime/npz_all/npz/layout"
CKPT_BASE="checkpoints/expansion"

# Collections to run (default all 3 new ones; override via CLI args)
if [[ $# -gt 0 ]]; then
    COLLECTIONS=("$@")
else
    COLLECTIONS=("xla/default" "nlp/random" "nlp/default")
fi

if [[ -f ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

mkdir -p "${CKPT_BASE}"
log "Spine A expansion — collections: ${COLLECTIONS[*]}"
log "30 runs max (3 coll × 5 seeds × 2 models). nlp collections dominate; expect days."

# ── Per-(collection, model, seed) run ────────────────────────────────────────
run_one() {
    local coll="$1" model="$2" seed="$3"
    local coll_safe="${coll//\//_}"
    local data_root="${DATA_BASE}/${coll}"
    local ckpt_dir="${CKPT_BASE}/${coll_safe}/${model}-seed${seed}"
    local run_name="${coll_safe}-${model}-seed${seed}"
    mkdir -p "${ckpt_dir}"

    # hardware model carries the two aux-loss weights; flat baseline ignores them
    local extra=""
    [[ "${model}" == "hardware" ]] && extra="--lp-weight ${LP_WEIGHT} --entropy-weight ${ENTROPY_WEIGHT}"

    local cmd="${PYTHON} train.py \
        --model ${model} \
        --data-root ${data_root} \
        --epochs ${EPOCHS} \
        --configs-per-batch ${CONFIGS_PER_BATCH} \
        --max-configs-per-graph ${MAX_CONFIGS_PER_GRAPH} \
        --num-workers ${NUM_WORKERS} \
        --seed ${seed} \
        --dropout 0.0 --lr 1e-3 --weight-decay 1e-4 --grad-clip 0.1 \
        --eval-every 5 ${extra} \
        --ckpt-dir ${ckpt_dir} \
        --project ${WANDB_PROJECT} --run-name ${run_name}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: ${cmd}"
    else
        log "RUN  ${run_name}"
        eval "${cmd}" 2>&1 | tee "${ckpt_dir}/run.log"
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for COLL in "${COLLECTIONS[@]}"; do
    COLL_SAFE="${COLL//\//_}"
    if [[ ! -d "${DATA_BASE}/${COLL}/train" ]]; then
        log "[SKIP] ${COLL} — no data at ${DATA_BASE}/${COLL}/train"
        continue
    fi
    log "════════ collection ${COLL} ════════"
    for SEED in "${SEEDS[@]}"; do
        run_one "${COLL}" "wide-baseline" "${SEED}"
        run_one "${COLL}" "hardware"      "${SEED}"
    done

    # ── Per-collection paired summary ────────────────────────────────────────
    [[ "${DRY_RUN:-0}" == "1" ]] && continue
    COLL="${COLL}" COLL_SAFE="${COLL_SAFE}" ${PYTHON} - <<'PYEOF' 2>/dev/null | tee "${CKPT_BASE}/${COLL_SAFE}/results_summary.txt"
import os, glob, statistics
from pathlib import Path
import torch
try:
    from scipy import stats as st
except Exception:
    st = None

coll      = os.environ["COLL"]
coll_safe = os.environ["COLL_SAFE"]
base      = Path("checkpoints/expansion") / coll_safe
seeds     = [42, 100, 123, 456, 789]
n_val     = len(glob.glob(f"data/predict-ai-model-runtime/npz_all/npz/layout/{coll}/valid/*.npz"))

def tau(model, s):
    p = base / f"{model}-seed{s}" / "best_model.pt"
    try:    return float(torch.load(p, map_location="cpu")["val_tau"])
    except Exception: return None

flat = [tau("wide-baseline", s) for s in seeds]
hw   = [tau("hardware",      s) for s in seeds]
pairs = [(f, h) for f, h in zip(flat, hw) if f is not None and h is not None]

print(f"Collection: {coll}   ({n_val} val graphs)")
print(f"  seeds: {seeds}")
for s, f, h in zip(seeds, flat, hw):
    if f is not None and h is not None:
        print(f"  seed {s:>3}:  flat {f:.4f}   hw {h:.4f}   Δ {h-f:+.4f}")
if pairs:
    F = [p[0] for p in pairs]; H = [p[1] for p in pairs]
    fm, fs = statistics.mean(F), (statistics.stdev(F) if len(F) > 1 else float('nan'))
    hm, hs = statistics.mean(H), (statistics.stdev(H) if len(H) > 1 else float('nan'))
    d = [h - f for f, h in pairs]
    dm = statistics.mean(d); ds = statistics.stdev(d) if len(d) > 1 else float('nan')
    print(f"\n  wide-baseline  τ = {fm:.4f} ± {fs:.4f}")
    print(f"  hardware       τ = {hm:.4f} ± {hs:.4f}")
    print(f"  paired Δτ (hw−flat) = {dm:+.4f} ± {ds:.4f}")
    if st is not None and len(d) > 1:
        t, p = st.ttest_rel(H, F)
        try: w, pw = st.wilcoxon(H, F); wtxt = f"  Wilcoxon p={pw:.3f}"
        except Exception: wtxt = ""
        print(f"  paired t-test p={p:.3f}{wtxt}")
        verdict = "NULL (no significant difference) — noise floor replicates" if p >= 0.05 \
                  else ("hardware SIGNIFICANTLY better" if dm > 0 else "flat SIGNIFICANTLY better")
        print(f"  ↳ {verdict}")
    print(f"\n  NOTE: {n_val} val graphs. Compare seed-std here vs xla:random (7 val) to test")
    print( "        whether the noise floor TIGHTENS with more evaluation graphs (Spine-A core).")
PYEOF
done

# ── Master roll-up ────────────────────────────────────────────────────────────
[[ "${DRY_RUN:-0}" == "1" ]] && { log "dry-run done"; exit 0; }
log "Building MASTER summary across collections…"
{
  echo "Spine A expansion — MASTER summary ($(date))"
  echo "Includes xla:random from the original Phase 2c/3 runs for reference."
  echo
  for f in "${CKPT_BASE}"/*/results_summary.txt; do
    [[ -f "$f" ]] && { cat "$f"; echo; echo "────────────────────────────────────────"; }
  done
} > "${CKPT_BASE}/MASTER_summary.txt"
log "Done. See ${CKPT_BASE}/MASTER_summary.txt"
