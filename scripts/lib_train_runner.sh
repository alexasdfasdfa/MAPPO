# shellcheck shell=bash
# Sourced by run_train_*.sh: start training so it keeps running after SSH/terminal disconnect.
#
# Uses: nohup, stdin detached from TTY (</dev/null), optional stdbuf for line-buffered logs,
# optional disown -h (bash), and absolute path to the conda env python when CONDA_PREFIX is set.
#
# Requires: LOG_FILE set; conda activate swE2 already done; cwd = REPO_ROOT.

run_train_with_nohup() {
  export PYTHONUNBUFFERED=1

  local py
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    py="${CONDA_PREFIX}/bin/python"
  else
    py=$(command -v python 2>/dev/null || echo python)
  fi

  local -a cmd=("${py}" "$@")
  local pid

  # Redirect stdin from /dev/null so the process is not tied to the terminal session.
  if command -v stdbuf >/dev/null 2>&1; then
    nohup stdbuf -oL -eL "${cmd[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
  else
    nohup "${cmd[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
  fi
  pid=$!

  # Bash: mark job so it is not sent SIGHUP when an interactive shell exits (belt-and-suspenders with nohup).
  if [[ -n "${BASH_VERSION:-}" ]] && disown -h "${pid}" 2>/dev/null; then
    :
  fi

  echo "Training PID=${pid} (nohup + detached stdin; survives closing this terminal / SSH)"
  echo "Log file: ${LOG_FILE}"
  echo "Follow:   tail -f \"${LOG_FILE}\""
  echo "Stop:     kill ${pid}"

  wait "${pid}"
}
