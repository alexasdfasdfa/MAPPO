#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p logs
mkdir -p .run
LOG_FILE="logs/tr_v3.nohup.$(date +%Y%m%d-%H%M%S).log"
PID_FILE=".run/tr_v3.pid"
AUTO_STOP_OLD="${AUTO_STOP_OLD:-1}"
STOP_WAIT_SECONDS="${STOP_WAIT_SECONDS:-8}"
TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-0}"

stop_old_if_needed() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 0
  fi

  old_pid="$(<"${PID_FILE}")"
  if [[ -z "${old_pid}" ]]; then
    rm -f "${PID_FILE}"
    return 0
  fi

  if ! kill -0 "${old_pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    return 0
  fi

  if [[ "${AUTO_STOP_OLD}" != "1" ]]; then
    echo "[info] Existing training PID=${old_pid} is running (AUTO_STOP_OLD=0)."
    echo "[info] Skip launching a new one."
    exit 0
  fi

  echo "[info] Stopping previous training PID=${old_pid} ..."
  kill "${old_pid}" 2>/dev/null || true
  for _ in $(seq 1 "${STOP_WAIT_SECONDS}"); do
    if ! kill -0 "${old_pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "[warn] PID=${old_pid} still alive, sending SIGKILL."
    kill -9 "${old_pid}" 2>/dev/null || true
  fi

  rm -f "${PID_FILE}"
}

start_training() {
  if [[ "${TRAIN_TIMEOUT_SECONDS}" -gt 0 ]]; then
    nohup bash -lc "timeout --signal=TERM --kill-after=20s ${TRAIN_TIMEOUT_SECONDS} bash scripts/tr_v3.sh \"$@\"" > "${LOG_FILE}" 2>&1 &
  else
    nohup bash scripts/tr_v3.sh "$@" > "${LOG_FILE}" 2>&1 &
  fi
  PID=$!
  echo "${PID}" > "${PID_FILE}"
}

stop_old_if_needed
start_training "$@"

echo "PID=${PID}"
echo "PID_FILE=${ROOT}/${PID_FILE}"
echo "LOG=${ROOT}/${LOG_FILE}"

