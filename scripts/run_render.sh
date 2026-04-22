#!/usr/bin/env bash
# Load conda, activate swE2, run MAPPO render.py from repo root; tee stdout/stderr to logs/.
#
# Usage:
#   cd /path/to/MAPPO
#   bash scripts/run_render.sh
#   bash scripts/run_render.sh /path/to/custom.log    # optional log path (must end with .log)
#   bash scripts/run_render.sh -- --model_dir ...     # extra args after -- go to render.py
#
# Examples:
#   bash scripts/run_render.sh -- --model_dir results/.../train/run1 --use_render
#   bash scripts/run_render.sh my_render.log -- --model_dir results/foo/train/run1
#
# Requires: conda, env "swE2" (see scripts/create_swE2_env.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x /opt/conda/bin/conda ]]; then
    export PATH="/opt/conda/bin:${PATH}"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install conda or add it to PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Missing ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate swE2

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
if [[ "${1:-}" =~ \.log$ ]]; then
  LOG_FILE="$1"
  shift
else
  LOG_FILE="${LOG_DIR}/render_$(date +%Y%m%d_%H%M%S).log"
fi

# Optional: strip a lone "--" so `bash run_render.sh -- --model_dir ...` works like train script
if [[ "${1:-}" == "--" ]]; then
  shift
fi

echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA_BASE}"
echo "Env:     swE2 (activated)"
echo "Log:     ${LOG_FILE}"
echo "Python:  $(command -v python)"
echo "--------"

set -o pipefail
python render.py "$@" 2>&1 | tee "${LOG_FILE}"

echo "Finished. Log saved to: ${LOG_FILE}"
