#!/usr/bin/env bash
# phase6b_swapper.sh — one-shot handover: when the last nlp_random job finishes,
# pause the Phase-6 sweep (watchdog first, then driver, then any raced-in
# nlp_default scope) and launch Phase 6b. Runs detached; exits after handover.
# Rationale + approval: session 2026-07-09 (reorder: 6b → 8c → nlp_default).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MARK="checkpoints/expansion/nlp_random/hardware-seed789/.done"
LOG="logs/capped/swapper.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] swapper armed — waiting for ${MARK}" >> "${LOG}"

# Wait up to 24 h (job is at ep29/50; expect ~2-4 h)
for i in $(seq 1 1440); do
    [[ -f "${MARK}" ]] && break
    sleep 60
done
if [[ ! -f "${MARK}" ]]; then
    echo "[$(ts)] TIMEOUT — marker never appeared; NOT swapping. Sweep left untouched." >> "${LOG}"
    exit 1
fi
echo "[$(ts)] marker present — beginning handover" >> "${LOG}"

# 1. watchdog FIRST (else it would restart the driver we're about to stop)
wd=$(pgrep -f "sweep_watchdog\.sh" | head -1)
[[ -n "${wd}" ]] && kill "${wd}" 2>/dev/null && echo "[$(ts)] watchdog ${wd} stopped" >> "${LOG}"

# 2. sweep driver
dr=$(pgrep -f "run_expansion_capped\.sh" | head -1)
[[ -n "${dr}" ]] && kill "${dr}" 2>/dev/null && echo "[$(ts)] sweep driver ${dr} stopped" >> "${LOG}"
sleep 3

# 3. any nlp_default scope that raced in between marker-write and our kill
for U in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep -o 'tpu-nlp_default[^ ]*'); do
    systemctl --user stop "${U}" 2>/dev/null && echo "[$(ts)] raced-in scope ${U} stopped" >> "${LOG}"
done
systemctl --user reset-failed 2>/dev/null || true
sleep 2

# 4. launch Phase 6b
setsid nohup bash run_phase6b.sh > logs/capped/phase6b_master.log 2>&1 < /dev/null &
echo "[$(ts)] Phase 6b launched (driver pid $!) — handover complete" >> "${LOG}"
