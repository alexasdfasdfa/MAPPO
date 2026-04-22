#!/usr/bin/env bash
# Same as run_train_undet_v2.sh plus undetermined v2 target exchange (pairwise swaps).
# Training runs under nohup (survives terminal close); see lib_train_runner.sh.
# CLI flag: --enable_undetermined_v2_exchange (see README.md "v2_exchange").
# Default shaping: reward shrinking fleet bottleneck max(dist2goal), sum(dist2goal), and
# swap-induced drop in total assigned distance (tune via undetermined_v2_exchange_*_scale).
#
# Usage:
#     cd /home/inno/MAPPO
#     bash scripts/run_train_undet_v2_exchange.sh
#     bash scripts/run_train_undet_v2_exchange.sh /path/to/run.log
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
  LOG_FILE="${LOG_DIR}/train_undet_v2_exchange_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA_BASE}"
echo "Env:     swE2 (activated)"
echo "Flags:   + --enable_undetermined_v2_exchange (v2 target exchange)"
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
  --enable_undetermined_v2_exchange \
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
  "$@"

echo "Finished. Log saved to: ${LOG_FILE}"
