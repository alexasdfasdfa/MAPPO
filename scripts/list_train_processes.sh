#!/usr/bin/env bash
# List MAPPO training Python processes (train.py) for the current user:
#   PID, elapsed time, CPU%, MEM%, RSS, full command line.
#
# Usage:
#   bash scripts/list_train_processes.sh
#   bash scripts/list_train_processes.sh --all-python   # all your python processes (noisy)
#
# Requires: ps (GNU coreutils preferred for etimes=pseconds)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USER_NAME="$(id -un)"
ALL_PY=0

for arg in "$@"; do
  case "$arg" in
    --all-python) ALL_PY=1 ;;
    -h|--help)
      sed -n '2,11p' "$0"
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
trap 'rm -f "${tmp}"' EXIT

if [[ "${ALL_PY}" -eq 1 ]]; then
  ps -u "${USER_NAME}" ww -o "${PS_FORMAT}" 2>/dev/null | grep -E '[p]ython|[P]ython' | grep -v list_train_processes.sh >"${tmp}" || true
else
  ps -u "${USER_NAME}" ww -o "${PS_FORMAT}" 2>/dev/null | grep -E '[p]ython.*train\.py|[P]ython.*train\.py' | grep -v list_train_processes.sh >"${tmp}" || true
fi

count="$(wc -l <"${tmp}" | tr -d ' ')"
if [[ "${count}" -eq 0 ]]; then
  echo "Matches:  0 process(es)."
  echo "Tip: training must show 'python ... train.py' in argv. Try: bash $0 --all-python"
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
echo "Stop:     kill <PID>        force: kill -9 <PID>"
echo "Logs:     tail -f ${REPO_ROOT}/logs/train_*.log"
