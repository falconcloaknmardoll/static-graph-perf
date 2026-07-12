#!/usr/bin/env bash
# sweep_watchdog.sh — self-healing monitor for the Phase 6 capped sweep.
# ─────────────────────────────────────────────────────────────────────────────
# Launched detached (setsid nohup) so it survives this terminal/session closing.
# Every CYCLE seconds it:
#   • records health to logs/capped/health.log and a glanceable STATUS.txt
#   • SELF-HEALS: if the sweep is incomplete, the driver process is dead, AND
#     nothing is mid-run, it relaunches run_expansion_capped.sh (idempotent —
#     the .done markers make it skip finished runs)
#   • flags the orphan-scope failure mode (should not recur with num-workers=0)
#   • exits when all 30 runs have a .done marker
#
# Does NOT survive a full machine reboot (it dies with everything else). For
# reboot resilience, add the crontab @reboot line printed by --print-cron.
#
# Usage:
#   setsid nohup bash scripts/sweep_watchdog.sh > logs/capped/watchdog.out 2>&1 < /dev/null &
#   bash scripts/sweep_watchdog.sh --print-cron     # show the reboot-proofing line
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CYCLE="${CYCLE:-1800}"          # 30 min
TOTAL=30
CKPT_BASE="checkpoints/expansion"
HEALTH="logs/capped/health.log"
STATUS="logs/capped/STATUS.txt"
DRIVER="run_expansion_capped.sh"
mkdir -p logs/capped

if [[ "${1:-}" == "--print-cron" ]]; then
    echo "# paste into 'crontab -e' to restart sweep + watchdog after a reboot:"
    echo "@reboot cd ${ROOT} && setsid nohup bash run_expansion_capped.sh > logs/capped/sweep_master.log 2>&1 < /dev/null & sleep 30; setsid nohup bash scripts/sweep_watchdog.sh > logs/capped/watchdog.out 2>&1 < /dev/null &"
    exit 0
fi

note() {  # best-effort desktop notification (may be a no-op when detached)
    command -v notify-send >/dev/null 2>&1 && \
        DISPLAY="${DISPLAY:-:0}" notify-send -u critical "TPUGraphs sweep" "$1" 2>/dev/null || true
}
ts() { date '+%Y-%m-%d %H:%M:%S'; }

restarts=0
echo "[$(ts)] watchdog START (cycle=${CYCLE}s, watching ${TOTAL}-run sweep)" >> "${HEALTH}"

while true; do
    done_n=$(find "${CKPT_BASE}" -name .done 2>/dev/null | wc -l | tr -d ' ')
    driver_pid=$(pgrep -f "${DRIVER}" | head -1)
    scopes=$(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep -c 'tpu-' || true)
    trainpy=$(pgrep -fc "train\.py" 2>/dev/null || true); trainpy=${trainpy:-0}
    avail=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo)
    fails=$(grep -c "FAIL " logs/capped/sweep_master.log 2>/dev/null || true); fails=${fails:-0}
    cur=$(grep -hoE "Epoch +[0-9]+/50" logs/capped/*.log 2>/dev/null | tail -1)
    state="OK"; action=""
    if [[ -n "${driver_pid}" ]]; then driver_str="alive (pid ${driver_pid})"; else driver_str="DOWN"; fi

    # ── completion ──────────────────────────────────────────────────────────
    if [[ "${done_n}" -ge "${TOTAL}" ]]; then
        printf 'SWEEP COMPLETE — %s/%s done at %s\n' "${done_n}" "${TOTAL}" "$(ts)" > "${STATUS}"
        echo "[$(ts)] COMPLETE ${done_n}/${TOTAL}" >> "${HEALTH}"
        note "Sweep COMPLETE: ${done_n}/${TOTAL} runs done."
        break
    fi

    # ── self-heal: driver dead + nothing running + work remaining ────────────
    if [[ -z "${driver_pid}" && "${scopes}" -eq 0 ]]; then
        state="DRIVER DOWN — restarting"
        setsid nohup bash run_expansion_capped.sh > logs/capped/sweep_master.log 2>&1 < /dev/null &
        restarts=$((restarts+1))
        action="relaunched driver (restart #${restarts})"
        echo "[$(ts)] SELF-HEAL: ${action}" >> "${HEALTH}"
        note "Sweep driver was down — watchdog restarted it (#${restarts})."
    # ── orphan guard (should not trigger with num-workers=0) ─────────────────
    elif [[ "${scopes}" -gt 1 ]]; then
        state="WARN: ${scopes} scopes active (orphan?)"
        echo "[$(ts)] WARN ${scopes} tpu- scopes active — possible orphan" >> "${HEALTH}"
        note "Warning: ${scopes} sweep scopes active (possible orphan accumulation)."
    fi

    # ── glanceable status + health line ──────────────────────────────────────
    {
        echo "Phase 6 sweep status — updated $(ts)"
        echo "  progress : ${done_n}/${TOTAL} runs complete"
        echo "  driver   : ${driver_str}"
        echo "  running  : ${cur:-<none>}   (scopes=${scopes}, train.py=${trainpy})"
        echo "  host RAM : ${avail} GiB available"
        echo "  failures : ${fails} FAIL line(s) in master log"
        echo "  watchdog : ${state}; restarts=${restarts}"
        [[ -n "${action}" ]] && echo "  last act : ${action}"
    } > "${STATUS}"
    echo "[$(ts)] done=${done_n}/${TOTAL} driver=${driver_pid:-DOWN} scopes=${scopes} trainpy=${trainpy} ramGiB=${avail} fails=${fails} ${cur:-} | ${state}" >> "${HEALTH}"

    sleep "${CYCLE}"
done
