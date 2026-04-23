#!/usr/bin/env bash
# Render with the same undetermined v2 + v2_exchange flags as run_train_undet_v2_exchange.sh
# (pair_mlp head, latent selector path, exchange shaping scales). Logs via tee (foreground).
#
# Usage:
#     cd /home/inno/MAPPO
#     bash scripts/run_render_undet_v2_exchange.sh -- --model_dir results/.../train/runN/models
#     bash scripts/run_render_undet_v2_exchange.sh /path/to/render.log -- --model_dir ...
#
# Extra args after -- are passed to render.py (e.g. --render_episodes 20 --use_render).
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
  LOG_FILE="${LOG_DIR}/render_undet_v2_exchange_$(date +%Y%m%d_%H%M%S).log"
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA_BASE}"
echo "Env:     swE2 (activated)"
echo "Flags:   match run_train_undet_v2_exchange.sh (+ --enable_undetermined_v2_exchange)"
echo "Log:     ${LOG_FILE}"
echo "Python:  $(command -v python)"
echo "--------"

set -o pipefail
python render.py \
  --train_font_pattern_length 10 \
  --train_font_pattern_policy all \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v2 \
  --enable_undetermined_v2_exchange \
  --undetermined_v2_exchange_accept_criterion fleet_m \
  --undetermined_v2_exchange_max_pairs_per_step 1 \
  --robot_initial_spawn_mode cluster_disk \
  --robot_init_cluster_radius_mode comm_vis_adaptive \
  --undetermined_v2_type2_formation_efficiency \
  --undetermined_v2_goal_slots 10 \
  --undet_v2_head_arch pair_mlp \
  --undetermined_target_embed_dim 32 \
  --undet_v2_pair_mlp_hidden 384 \
  --undet_v2_target_latent_model_dir "../selector_n15.pt" \
  --undet_v2_latent_train_mode finetune_all \
  --undetermined_v2_exchange_bottleneck_shaping_scale 0.06 \
  --undetermined_v2_exchange_team_dist_shaping_scale 0.04 \
  --undetermined_v2_exchange_swap_bonus_scale 0.10 \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo "Finished. Log saved to: ${LOG_FILE}"
