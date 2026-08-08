#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 2 ]]; then
    echo "usage: $0 RESULT_FILE COMMAND [ARG ...]" >&2
    exit 2
fi

RESULT_FILE="$1"
shift
RESULT_TMP="${RESULT_FILE}.tmp.$$"

write_result() {
    local state="$1"
    local exit_code="$2"
    {
        printf 'state=%s\n' "${state}"
        printf 'exit_code=%s\n' "${exit_code}"
        printf 'pid=%s\n' "$$"
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    } > "${RESULT_TMP}"
    mv -f "${RESULT_TMP}" "${RESULT_FILE}"
}

on_exit() {
    local exit_code=$?
    trap - EXIT
    write_result finished "${exit_code}"
    exit "${exit_code}"
}

trap on_exit EXIT
write_result running -1
"$@"
