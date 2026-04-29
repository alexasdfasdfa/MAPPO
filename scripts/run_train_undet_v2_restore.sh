#!/usr/bin/env bash
# Restore MAPPO original undetermined v2 training settings (model + flow).
#
# Usage:
#   cd /home/wangdx_lab/cse12211818/MAPPO
#   bash scripts/run_train_undet_v2_restore.sh

set -euo pipefail
date
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
  LOG_FILE="${LOG_DIR}/train_undet_v2_restore_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA_BASE}"
echo "Env:     swE2 (activated)"
echo "Log:     ${LOG_FILE}"
echo "Python:  $(command -v python)"
echo "--------"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib_train_runner.sh"
set -o pipefail
run_train_with_nohup train.py \
  --train_font_pattern_length 10 \
  --train_font_pattern_policy all \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v2 \
  --robot_initial_spawn_mode cluster_disk \
  --robot_init_cluster_radius_mode comm_vis_adaptive \
  --undetermined_v2_type2_formation_efficiency \
  --undetermined_v2_goal_slots 10 \
  --undet_v2_head_arch pair_mlp \
  --undetermined_target_embed_dim 32 \
  --undet_v2_pair_mlp_hidden 384 \
  "$@"

echo "Finished. Log saved to: ${LOG_FILE}"
