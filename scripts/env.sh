#!/usr/bin/env bash
# Initialize conda + activate swE2; install scipy into that env (not user site).
#
# From MAPPO repo root:
#   source scripts/env.sh    # recommended: swE2 stays active in this shell
#   bash scripts/env.sh      # one-shot in a subshell (activate does not persist)
#
# If you only see "CondaError: Run 'conda init'", you ran conda activate without
# loading conda.sh first — this file does that for you.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x /opt/conda/bin/conda ]]; then
    export PATH="/opt/conda/bin:${PATH}"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  return 2 2>/dev/null || exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate swE2

PY="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Expected ${PY} after conda activate swE2" >&2
  return 1 2>/dev/null || exit 1
fi

echo "Using: $("${PY}" -V) (${PY})"
"${PY}" -m pip install "scipy>=1.10"

"${PY}" -c "import scipy; from scipy.optimize import linear_sum_assignment; print('scipy OK', scipy.__version__)"
