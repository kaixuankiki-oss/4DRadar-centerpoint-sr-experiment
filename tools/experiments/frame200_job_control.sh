#!/usr/bin/env bash
set -euo pipefail

# Persistent control for the Frame200 SR/CenterPoint experiment. Prefer a
# lingering user-level systemd service so the command survives SSH/Codex
# disconnects and can be paused/resumed together with all of its children.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.experiment_control/frame200_sr"
PID_FILE="${STATE_DIR}/job.pid"
COMMAND_FILE="${STATE_DIR}/job.command"
LOG_FILE="${STATE_DIR}/job.log"
RESULT_FILE="${STATE_DIR}/job.result"
BACKEND_FILE="${STATE_DIR}/job.backend"
PAUSE_FILE="${STATE_DIR}/PAUSE"
STOP_FILE="${STATE_DIR}/STOP"
RUNNER="${SCRIPT_DIR}/frame200_job_runner.sh"
UNIT_NAME="frame200-sr-iteration.service"

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

read_backend() {
    if [[ -s "${BACKEND_FILE}" ]]; then
        read -r JOB_BACKEND < "${BACKEND_FILE}"
    else
        JOB_BACKEND="legacy"
    fi
}

systemd_user_available() {
    systemctl --user show-environment >/dev/null 2>&1
}

systemd_active() {
    systemctl --user is-active --quiet "${UNIT_NAME}"
}

systemd_main_pid() {
    systemctl --user show "${UNIT_NAME}" --property=MainPID --value 2>/dev/null || true
}

show_status() {
    read_backend
    if [[ "${JOB_BACKEND}" == "systemd" ]]; then
        if ! systemd_user_available; then
            echo "status=unavailable backend=systemd (run from the host user session)"
        elif systemd_active; then
            JOB_PID="$(systemd_main_pid)"
            local stat
            stat="$(ps -o stat= -p "${JOB_PID}" 2>/dev/null | tr -d ' ' || true)"
            if [[ "${stat}" == *T* ]]; then
                echo "status=paused backend=systemd pid=${JOB_PID}"
            else
                echo "status=running backend=systemd pid=${JOB_PID}"
            fi
            ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,cmd -p "${JOB_PID}" || true
        elif [[ -e "${STOP_FILE}" ]]; then
            echo "status=stopped backend=systemd"
        elif [[ -s "${RESULT_FILE}" ]]; then
            # shellcheck disable=SC1090
            source "${RESULT_FILE}"
            if [[ "${state:-}" == "finished" && "${exit_code:-1}" -eq 0 ]]; then
                echo "status=completed backend=systemd exit_code=${exit_code}"
            elif [[ "${state:-}" == "finished" ]]; then
                echo "status=failed backend=systemd exit_code=${exit_code:-unknown}"
            else
                echo "status=idle backend=systemd last_state=${state:-unknown}"
            fi
        else
            echo "status=idle backend=systemd"
        fi
        echo "unit=${UNIT_NAME}"
        echo "log=${LOG_FILE}"
        [[ -s "${COMMAND_FILE}" ]] && echo "command=$(<"${COMMAND_FILE}")"
        return
    fi

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
        if systemd_user_available; then
            if systemd_active; then
                echo "A managed systemd job is already active." >&2
                exit 1
            fi
            JOB_BACKEND="systemd"
        else
            JOB_BACKEND="legacy"
            read_pid
            if is_alive; then
                echo "A managed job is already active (pid=${JOB_PID})." >&2
                exit 1
            fi
        fi
        rm -f "${PAUSE_FILE}" "${STOP_FILE}"
        rm -f "${RESULT_FILE}"
        printf '%q ' "$@" > "${COMMAND_FILE}"
        printf '\n' >> "${COMMAND_FILE}"
        : > "${LOG_FILE}"
        printf '%s\n' "${JOB_BACKEND}" > "${BACKEND_FILE}"
        if [[ "${JOB_BACKEND}" == "systemd" ]]; then
            systemctl --user reset-failed "${UNIT_NAME}" >/dev/null 2>&1 || true
            systemd-run --user --quiet --collect --service-type=exec \
                --unit="${UNIT_NAME}" \
                --property=KillMode=control-group \
                --property=TimeoutStopSec=30 \
                --property="WorkingDirectory=${REPO_ROOT}" \
                --property="StandardOutput=append:${LOG_FILE}" \
                --property="StandardError=append:${LOG_FILE}" \
                "${RUNNER}" "${RESULT_FILE}" "$@"
            JOB_PID="$(systemd_main_pid)"
        else
            # Fallback for hosts without a user systemd manager.
            nohup setsid --wait "${RUNNER}" "${RESULT_FILE}" "$@" \
                </dev/null >> "${LOG_FILE}" 2>&1 &
            JOB_PID=$!
            disown "${JOB_PID}" 2>/dev/null || true
        fi
        printf '%s\n' "${JOB_PID}" > "${PID_FILE}"
        sleep 0.2
        if [[ "${JOB_BACKEND}" == "systemd" ]] && ! systemd_active; then
            echo "Managed systemd command exited during startup; see ${LOG_FILE}." >&2
            exit 1
        elif [[ "${JOB_BACKEND}" == "legacy" ]] && ! is_alive; then
            echo "Managed command exited during startup; see ${LOG_FILE}." >&2
            exit 1
        fi
        echo "started backend=${JOB_BACKEND} pid=${JOB_PID} log=${LOG_FILE}"
        ;;
    pause)
        touch "${PAUSE_FILE}"
        read_backend
        if [[ "${JOB_BACKEND}" == "systemd" ]] && systemd_active; then
            systemctl --user kill --kill-whom=all --signal=STOP "${UNIT_NAME}"
            echo "paused unit=${UNIT_NAME} (entire service cgroup)"
        else
            read_pid
            if is_alive; then
                kill -STOP -- "-${JOB_PID}"
                echo "paused pid=${JOB_PID} (entire process group)"
            else
                echo "pause requested; no active managed process"
            fi
        fi
        ;;
    resume)
        rm -f "${PAUSE_FILE}" "${STOP_FILE}"
        read_backend
        if [[ "${JOB_BACKEND}" == "systemd" ]] && systemd_active; then
            systemctl --user kill --kill-whom=all --signal=CONT "${UNIT_NAME}"
            echo "resumed unit=${UNIT_NAME}"
        else
            read_pid
            if is_alive; then
                kill -CONT -- "-${JOB_PID}"
                echo "resumed pid=${JOB_PID}"
            else
                echo "resume flag cleared; no active managed process"
            fi
        fi
        ;;
    stop)
        touch "${STOP_FILE}"
        rm -f "${PAUSE_FILE}"
        read_backend
        if [[ "${JOB_BACKEND}" == "systemd" ]] && systemd_active; then
            systemctl --user kill --kill-whom=all --signal=CONT "${UNIT_NAME}" 2>/dev/null || true
            systemctl --user stop "${UNIT_NAME}"
            echo "stopped unit=${UNIT_NAME}"
        else
            read_pid
            if is_alive; then
                # Continue a paused group first so it can handle SIGTERM cleanly.
                kill -CONT -- "-${JOB_PID}" 2>/dev/null || true
                kill -TERM -- "-${JOB_PID}"
                echo "graceful stop requested for pid=${JOB_PID}"
            else
                echo "stop requested; no active managed process"
            fi
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
        read_backend
        if [[ "${JOB_BACKEND}" == "systemd" ]] && systemd_user_available && systemd_active; then
            echo "Cannot clear state while ${UNIT_NAME} is active." >&2
            exit 1
        fi
        read_pid
        if [[ "${JOB_BACKEND}" != "systemd" ]] && is_alive; then
            echo "Cannot clear state while pid=${JOB_PID} is active." >&2
            exit 1
        fi
        rm -f "${PID_FILE}" "${COMMAND_FILE}" "${LOG_FILE}" "${RESULT_FILE}" \
            "${BACKEND_FILE}" "${PAUSE_FILE}" "${STOP_FILE}"
        echo "cleared ${STATE_DIR}"
        ;;
    *)
        echo "usage: $0 {start COMMAND...|pause|resume|stop|status|log [LINES]|clear}" >&2
        exit 2
        ;;
esac
