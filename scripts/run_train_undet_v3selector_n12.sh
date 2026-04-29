#!/usr/bin/env bash
# Launch MAPPO v3 branch with "pretrained selector -> motion policy" pipeline.
# This script targets undetermined v3 branch (AttnComm + attn_undetermined_goal).
#
# Usage:
#   cd /home/wangdx_lab/cse12211818/MAPPO
#   bash scripts/run_train_undet_v3selector_n12.sh
#   # or: bash scripts/run_train_undet_v3selector_n12.sh /path/to/log.log
#
# Optional overrides:
#   export LATENT_TRAIN_MODE=motion_only   # motion_only | finetune_all

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
  LOG_FILE="${LOG_DIR}/train_undet_v3selector_n12_$(date +%Y%m%d_%H%M%S).log"
fi

LATENT_MODEL_DIR="../undet_v3_target_latent_decoupled_rank/checkpoints_decoupled_equal/v3decoupled_equal_n12.pt"
LATENT_TRAIN_MODE="${LATENT_TRAIN_MODE:-motion_only}"

if [[ ! -f "${LATENT_MODEL_DIR}" ]]; then
  echo "[error] latent checkpoint not found: ${LATENT_MODEL_DIR}" >&2
  exit 1
fi

echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA_BASE}"
echo "Env:     swE2 (activated)"
echo "Log:     ${LOG_FILE}"
echo "Python:  $(command -v python)"
echo "Latent:  ${LATENT_MODEL_DIR}"
echo "Mode:    ${LATENT_TRAIN_MODE}"
echo "--------"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib_train_runner.sh"
set -o pipefail
run_train_with_nohup train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  --robot_initial_spawn_mode random_box \
  --undetermined_v2_type2_formation_efficiency \
  --undetermined_v2_goal_slots 10 \
  --undet_v2_head_arch dot_product \
  --undet_v3_head_arch decoupled_rank_compat \
  --undetermined_target_embed_dim 32 \
  --undet_v3_target_latent_model_dir "${LATENT_MODEL_DIR}" \
  --undet_v3_latent_train_mode "${LATENT_TRAIN_MODE}" \
  "$@"

echo "Finished. Log saved to: ${LOG_FILE}"
