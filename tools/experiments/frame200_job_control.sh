#!/usr/bin/env bash
set -euo pipefail

# Persistent control for the Frame200 SR/CenterPoint experiment.  A managed
# command is started in a separate session/process group, so it survives an
# SSH disconnect and can be paused/resumed together with all of its children.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.experiment_control/frame200_sr"
PID_FILE="${STATE_DIR}/job.pid"
COMMAND_FILE="${STATE_DIR}/job.command"
LOG_FILE="${STATE_DIR}/job.log"
PAUSE_FILE="${STATE_DIR}/PAUSE"
STOP_FILE="${STATE_DIR}/STOP"

mkdir -p "${STATE_DIR}"

read_pid() {
    if [[ -s "${PID_FILE}" ]]; then
        read -r JOB_PID < "${PID_FILE}"
    else
        JOB_PID=""
    fi
}

is_alive() {
    [[ -n "${JOB_PID:-}" ]] && kill -0 "${JOB_PID}" 2>/dev/null
}

show_status() {
    read_pid
    if is_alive; then
        local stat
        stat="$(ps -o stat= -p "${JOB_PID}" | tr -d ' ')"
        if [[ "${stat}" == *T* ]]; then
            echo "status=paused pid=${JOB_PID}"
        else
            echo "status=running pid=${JOB_PID}"
        fi
        ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,cmd -p "${JOB_PID}"
    elif [[ -e "${STOP_FILE}" ]]; then
        echo "status=stopped"
    elif [[ -e "${PAUSE_FILE}" ]]; then
        echo "status=pause_requested (no active managed process)"
    else
        echo "status=idle"
    fi
    echo "log=${LOG_FILE}"
    [[ -s "${COMMAND_FILE}" ]] && echo "command=$(<"${COMMAND_FILE}")"
}

action="${1:-status}"
case "${action}" in
    start)
        shift
        if [[ "$#" -eq 0 ]]; then
            echo "usage: $0 start COMMAND [ARG ...]" >&2
            exit 2
        fi
        read_pid
        if is_alive; then
            echo "A managed job is already active (pid=${JOB_PID})." >&2
            exit 1
        fi
        rm -f "${PAUSE_FILE}" "${STOP_FILE}"
        printf '%q ' "$@" > "${COMMAND_FILE}"
        printf '\n' >> "${COMMAND_FILE}"
        : > "${LOG_FILE}"
        setsid --wait "$@" >> "${LOG_FILE}" 2>&1 &
        JOB_PID=$!
        printf '%s\n' "${JOB_PID}" > "${PID_FILE}"
        sleep 0.2
        if ! is_alive; then
            echo "Managed command exited during startup; see ${LOG_FILE}." >&2
            exit 1
        fi
        echo "started pid=${JOB_PID} log=${LOG_FILE}"
        ;;
    pause)
        touch "${PAUSE_FILE}"
        read_pid
        if is_alive; then
            kill -STOP -- "-${JOB_PID}"
            echo "paused pid=${JOB_PID} (entire process group)"
        else
            echo "pause requested; no active managed process"
        fi
        ;;
    resume)
        rm -f "${PAUSE_FILE}" "${STOP_FILE}"
        read_pid
        if is_alive; then
            kill -CONT -- "-${JOB_PID}"
            echo "resumed pid=${JOB_PID}"
        else
            echo "resume flag cleared; no active managed process"
        fi
        ;;
    stop)
        touch "${STOP_FILE}"
        rm -f "${PAUSE_FILE}"
        read_pid
        if is_alive; then
            # Continue a paused group first so it can handle SIGTERM cleanly.
            kill -CONT -- "-${JOB_PID}" 2>/dev/null || true
            kill -TERM -- "-${JOB_PID}"
            echo "graceful stop requested for pid=${JOB_PID}"
        else
            echo "stop requested; no active managed process"
        fi
        ;;
    status)
        show_status
        ;;
    log)
        lines="${2:-80}"
        tail -n "${lines}" "${LOG_FILE}" 2>/dev/null || true
        ;;
    clear)
        read_pid
        if is_alive; then
            echo "Cannot clear state while pid=${JOB_PID} is active." >&2
            exit 1
        fi
        rm -f "${PID_FILE}" "${COMMAND_FILE}" "${LOG_FILE}" "${PAUSE_FILE}" "${STOP_FILE}"
        echo "cleared ${STATE_DIR}"
        ;;
    *)
        echo "usage: $0 {start COMMAND...|pause|resume|stop|status|log [LINES]|clear}" >&2
        exit 2
        ;;
esac
