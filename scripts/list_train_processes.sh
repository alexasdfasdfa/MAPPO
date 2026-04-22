#!/usr/bin/env bash
# List MAPPO training Python processes for the current user:
#   PID, elapsed time, CPU%, MEM%, RSS, command line from ps.
#
# train.py calls setproctitle (see train.py), so ps often shows only "@<user_name>"
# with no "python" / "train.py" in argv. On Linux we also detect PIDs whose cwd is
# this repo and whose executable is python* (see collect_pids below).
#
# Usage:
#   bash scripts/list_train_processes.sh
#   bash scripts/list_train_processes.sh --with-workers # include DataLoader / worker @title PIDs
#   bash scripts/list_train_processes.sh --all-python   # all your python processes (noisy)
#
# Requires: ps (GNU coreutils preferred for etimes=pseconds); Linux /proc for argv hiding

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USER_NAME="$(id -un)"
ALL_PY=0
WITH_WORKERS=0

for arg in "$@"; do
  case "$arg" in
    --all-python) ALL_PY=1 ;;
    --with-workers) WITH_WORKERS=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
  esac
done

# GNU ps: etimes = elapsed wall time in seconds; fallback etime=[[dd-]hh:]mm:ss
PS_FORMAT="pid=,etimes=,pcpu=,pmem=,rss=,args="
if ! ps -p 1 -o etimes= >/dev/null 2>&1; then
  PS_FORMAT="pid=,etime=,pcpu=,pmem=,rss=,args="
fi

echo "User:     ${USER_NAME}"
echo "MAPPO:    ${REPO_ROOT}"

tmp="$(mktemp)"
tmp_pids="$(mktemp)"
trap 'rm -f "${tmp}" "${tmp_pids}"' EXIT

collect_train_pids() {
  : >"${tmp_pids}"
  if [[ "${ALL_PY}" -eq 1 ]]; then
    ps -u "${USER_NAME}" ww -o "${PS_FORMAT}" 2>/dev/null | grep -E '[p]ython|[P]ython' | grep -v list_train_processes.sh | awk '{print $1}' >>"${tmp_pids}" || true
    sort -un "${tmp_pids}" -o "${tmp_pids}"
    return 0
  fi

  # (1) argv still mentions train.py (no setproctitle yet, or short run)
  ps -u "${USER_NAME}" ww -o "${PS_FORMAT}" 2>/dev/null \
    | grep -E '[p]ython.*train\.py|[P]ython.*train\.py' \
    | grep -v list_train_processes.sh \
    | awk '{print $1}' >>"${tmp_pids}" || true

  # (2) Linux: python with cwd=this repo (argv hidden as "@user" by setproctitle in train.py).
  #    Default: only session roots (parent cmdline mentions train.py or run_train*.sh).
  #    --with-workers: every matching python in the repo (many @title workers).
  if [[ -d /proc ]]; then
    local myuid repo_canon pid proc uid exe bn cwd ppid pline first pexe pbn
    myuid="$(id -u)"
    repo_canon="$(readlink -f "${REPO_ROOT}")"
    for proc in /proc/[0-9]*; do
      [[ -d "${proc}" ]] || continue
      pid="${proc##*/}"
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      uid="$(stat -c '%u' "${proc}" 2>/dev/null)" || continue
      [[ "${uid}" == "${myuid}" ]] || continue
      exe="$(readlink -f "${proc}/exe" 2>/dev/null)" || continue
      bn="$(basename "${exe}")"
      [[ "${bn}" == python* ]] || continue
      cwd="$(readlink -f "${proc}/cwd" 2>/dev/null)" || continue
      [[ "${cwd}" == "${repo_canon}" ]] || continue
      if [[ "${WITH_WORKERS}" -eq 1 ]]; then
        printf '%s\n' "${pid}"
        continue
      fi
      ppid="$(awk '/^PPid:/{print $2}' "${proc}/status" 2>/dev/null)" || continue
      [[ -n "${ppid}" && "${ppid}" != "0" ]] || continue
      if [[ -r "/proc/${ppid}/cmdline" ]]; then
        pline="$(tr '\0' ' ' <"/proc/${ppid}/cmdline" 2>/dev/null)"
        if [[ "${pline}" == *train.py* || "${pline}" == *run_train*.sh* ]]; then
          printf '%s\n' "${pid}"
          continue
        fi
      fi
      # setproctitle("@user"): workers stay under another python; mains sit under shell/nohup/etc.
      first="$(tr '\0' ' ' <"${proc}/cmdline" 2>/dev/null | awk '{print $1}')"
      if [[ "${first}" == @* ]]; then
        pexe="$(readlink -f "/proc/${ppid}/exe" 2>/dev/null)" || continue
        pbn="$(basename "${pexe}")"
        if [[ "${pbn}" != python* ]]; then
          printf '%s\n' "${pid}"
        fi
      fi
    done >>"${tmp_pids}" || true
  fi

  sort -un "${tmp_pids}" -o "${tmp_pids}"
}

build_ps_lines_for_pids() {
  : >"${tmp}"
  local pid line
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    line="$(ps -p "${pid}" ww -o "${PS_FORMAT}" 2>/dev/null)"
    [[ -z "${line// }" ]] && continue
    printf '%s\n' "${line}" >>"${tmp}"
  done <"${tmp_pids}"
}

collect_train_pids
build_ps_lines_for_pids

count="$(wc -l <"${tmp}" | tr -d ' ')"
if [[ "${count}" -eq 0 ]]; then
  echo "Matches:  0 process(es)."
  if [[ "${ALL_PY}" -eq 1 ]]; then
    echo "Tip: no python processes for user ${USER_NAME} (see ps -u ${USER_NAME} ww)."
  else
    echo "Tip: no train.py argv match and no python with cwd=${REPO_ROOT} (Linux /proc). Try: bash $0 --all-python"
  fi
  exit 0
fi

echo "Matches:  ${count} process(es)"
echo "--------------------------------------------------------------------------------"
printf '%-8s %-12s %-6s %-6s %-10s %s\n' "PID" "ELAPSED" "%CPU" "%MEM" "RSS(KB)" "COMMAND"
echo "--------------------------------------------------------------------------------"

while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue
  pid="$(echo "${line}" | awk '{print $1}')"
  et="$(echo "${line}" | awk '{print $2}')"
  cpu="$(echo "${line}" | awk '{print $3}')"
  mem="$(echo "${line}" | awk '{print $4}')"
  rss="$(echo "${line}" | awk '{print $5}')"
  cmd="$(echo "${line}" | awk '{for(i=6;i<=NF;i++) printf "%s%s", $i, (i<NF?" ":"")}')"
  [[ -z "${pid}" ]] && continue
  printf '%-8s %-12s %-6s %-6s %-10s %s\n' "${pid}" "${et}" "${cpu}" "${mem}" "${rss}" "${cmd}"
done <"${tmp}"

echo "--------------------------------------------------------------------------------"
echo "More PIDs: bash $0 --with-workers   (workers share the same @user argv from train.py)"
echo "Stop:     kill <PID>        force: kill -9 <PID>"
echo "Logs:     tail -f ${REPO_ROOT}/logs/train_*.log"
