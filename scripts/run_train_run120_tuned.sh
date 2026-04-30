#!/usr/bin/env bash
# Tuned training recipe for improving:
# - pair_success_rate / episode_all_agents_success_rate
# - mean_path_length
# - mean_all_agents_formation_step
#
# Key strategy:
# - Do NOT use "freeze selector then finetune selector" as default.
# - Train motion + selector jointly from early stage with reward curriculum.
#
# Usage:
#   cd /home/wangdx_lab/cse12211818/MAPPO
#   bash scripts/run_train_run120_tuned.sh
#   TARGET_EXCHANGE_ENABLE=0 bash scripts/run_train_run120_tuned.sh
#
# Optional first arg: custom log file (*.log)

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
  LOG_FILE="${LOG_DIR}/train_run120_tuned_$(date +%Y%m%d_%H%M%S).log"
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib_train_runner.sh"

LATENT_MODEL_DIR="${LATENT_MODEL_DIR:-../undet_v3_target_latent_decoupled_rank/checkpoints_decoupled_equal/v3decoupled_equal_n11 copy.pt}"
if [[ ! -f "${LATENT_MODEL_DIR}" ]]; then
  echo "[error] latent checkpoint not found: ${LATENT_MODEL_DIR}" >&2
  exit 1
fi

TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-1}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi

# Selector training mode:
# - finetune_all (recommended): joint train selector + motion from the start.
# - motion_only             : freeze selector head (not recommended by recent reward curves).
LATENT_TRAIN_MODE="${LATENT_TRAIN_MODE:-finetune_all}"
if [[ "${LATENT_TRAIN_MODE}" == "motion_only" ]]; then
  echo "[warn] LATENT_TRAIN_MODE=motion_only may hurt final performance based on run120/run121 reward curves." >&2
fi

# Keep run120-style no-attn-tail branch for compatibility, but restore target history.
V3_DISABLE_ATTN_TAIL="${V3_DISABLE_ATTN_TAIL:-1}"
V3_DISABLE_PREV_TARGET="${V3_DISABLE_PREV_TARGET:-0}"
V3_DISABLE_ATTN_TAIL_FLAG=""
if [[ "${V3_DISABLE_ATTN_TAIL}" == "1" ]]; then
  V3_DISABLE_ATTN_TAIL_FLAG="--undetermined_v3_disable_attn_tail_for_motion"
fi
V3_DISABLE_PREV_TARGET_FLAG=""
if [[ "${V3_DISABLE_PREV_TARGET}" == "1" ]]; then
  V3_DISABLE_PREV_TARGET_FLAG="--undetermined_v3_disable_prev_target_for_motion"
fi

# Improve sample efficiency / stability (favor faster early reward rise).
NUM_MINI_BATCH="${NUM_MINI_BATCH:-128}"
ENTROPY_COEF="${ENTROPY_COEF:-0.010}"

# Reward shaping tuned toward faster/shorter successful formation.
V3_REWARD_M_DROP_SCALE="${V3_REWARD_M_DROP_SCALE:-0.85}"
V3_REWARD_S_DROP_SCALE="${V3_REWARD_S_DROP_SCALE:-0.28}"
V3_REWARD_TRAVEL_PENALTY_SCALE="${V3_REWARD_TRAVEL_PENALTY_SCALE:-0.03}"
V3_SELECTOR_UNIQUE_BONUS_SCALE="${V3_SELECTOR_UNIQUE_BONUS_SCALE:-0.15}"
V3_SELECTOR_PENDING_PENALTY_SCALE="${V3_SELECTOR_PENDING_PENALTY_SCALE:-0.25}"
V3_SELECTOR_DUPLICATE_PENALTY_SCALE="${V3_SELECTOR_DUPLICATE_PENALTY_SCALE:-0.25}"
V3_SELECTOR_PROGRESS_BONUS_SCALE="${V3_SELECTOR_PROGRESS_BONUS_SCALE:-0.35}"
UNDET_V2_SL_DENSE_SCALE="${UNDET_V2_SL_DENSE_SCALE:-0.35}"
UNDET_V2_SL_DELTA_SCALE="${UNDET_V2_SL_DELTA_SCALE:-0.85}"
UNDET_V2_SL_SUCCESS_SCALE="${UNDET_V2_SL_SUCCESS_SCALE:-3.00}"
UNDET_V2_SL_SUCCESS_THRESHOLD="${UNDET_V2_SL_SUCCESS_THRESHOLD:-0.97}"
ND_ARRIVAL_REWARD="${ND_ARRIVAL_REWARD:-1.00}"
ND_GOAL_TERMINAL_REWARD="${ND_GOAL_TERMINAL_REWARD:-2.20}"

# Reward/curriculum mode:
# - joint_selector_early (default): selector participates earlier, avoids late-only selector recovery.
# - motion_first               : closer to old schedule.
REWARD_TRAIN_MODE="${REWARD_TRAIN_MODE:-joint_selector_early}"
V3_CURRICULUM_ENABLE="${V3_CURRICULUM_ENABLE:-1}"
if [[ "${REWARD_TRAIN_MODE}" == "motion_first" ]]; then
  V3_CURRICULUM_MOTION_PHASE_RATIO="${V3_CURRICULUM_MOTION_PHASE_RATIO:-0.45}"
  V3_CURRICULUM_MOTION_REWARD_BOOST="${V3_CURRICULUM_MOTION_REWARD_BOOST:-2.0}"
  V3_CURRICULUM_SELECTOR_REWARD_EARLY="${V3_CURRICULUM_SELECTOR_REWARD_EARLY:-0.25}"
  V3_CURRICULUM_SELECTOR_REWARD_LATE="${V3_CURRICULUM_SELECTOR_REWARD_LATE:-1.50}"
  V3_CURRICULUM_SELECTOR_KL_EARLY="${V3_CURRICULUM_SELECTOR_KL_EARLY:-0.18}"
  V3_CURRICULUM_SELECTOR_KL_LATE="${V3_CURRICULUM_SELECTOR_KL_LATE:-1.60}"
else
  V3_CURRICULUM_MOTION_PHASE_RATIO="${V3_CURRICULUM_MOTION_PHASE_RATIO:-0.20}"
  V3_CURRICULUM_MOTION_REWARD_BOOST="${V3_CURRICULUM_MOTION_REWARD_BOOST:-1.8}"
  V3_CURRICULUM_SELECTOR_REWARD_EARLY="${V3_CURRICULUM_SELECTOR_REWARD_EARLY:-0.90}"
  V3_CURRICULUM_SELECTOR_REWARD_LATE="${V3_CURRICULUM_SELECTOR_REWARD_LATE:-1.40}"
  V3_CURRICULUM_SELECTOR_KL_EARLY="${V3_CURRICULUM_SELECTOR_KL_EARLY:-0.70}"
  V3_CURRICULUM_SELECTOR_KL_LATE="${V3_CURRICULUM_SELECTOR_KL_LATE:-1.35}"
fi
V3_CURRICULUM_ENABLE_FLAG=""
if [[ "${V3_CURRICULUM_ENABLE}" == "1" ]]; then
  V3_CURRICULUM_ENABLE_FLAG="--undetermined_v3_curriculum_enable"
fi

echo "Repo:    ${REPO_ROOT}"
echo "Env:     swE2 (activated)"
echo "Log:     ${LOG_FILE}"
echo "Python:  $(command -v python)"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "V3_DISABLE_ATTN_TAIL=${V3_DISABLE_ATTN_TAIL}"
echo "V3_DISABLE_PREV_TARGET=${V3_DISABLE_PREV_TARGET}"
echo "LATENT_TRAIN_MODE=${LATENT_TRAIN_MODE}"
echo "REWARD_TRAIN_MODE=${REWARD_TRAIN_MODE}"
echo "--------"

set -o pipefail
run_train_with_nohup train.py \
  --experiment_name run120_tuned_efficiency \
  --train_font_pattern_length 10 \
  --train_font_pattern_policy all \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
  ${V3_DISABLE_ATTN_TAIL_FLAG} \
  ${V3_DISABLE_PREV_TARGET_FLAG} \
  ${V3_CURRICULUM_ENABLE_FLAG} \
  --randomize_robot_initial_positions \
  --robot_initial_spawn_mode random_box \
  --undetermined_v2_type2_formation_efficiency \
  --undetermined_v2_goal_slots 10 \
  --undetermined_v3_exchange_max_neighbors 10 \
  --undetermined_v3_comm_ally_slots 10 \
  --undetermined_v3_comm_human_slots 4 \
  --undetermined_obs_goal_radius 5.0 \
  --undetermined_comm_radius 5.0 \
  --undet_v2_head_arch dot_product \
  --undet_v3_head_arch decoupled_rank_compat \
  --undet_v3_latent_p_max_neighbors 10 \
  --undetermined_target_embed_dim 32 \
  --undetermined_v3_reward_enable \
  --undetermined_v3_reward_m_drop_scale "${V3_REWARD_M_DROP_SCALE}" \
  --undetermined_v3_reward_s_drop_scale "${V3_REWARD_S_DROP_SCALE}" \
  --undetermined_v3_reward_travel_penalty_scale "${V3_REWARD_TRAVEL_PENALTY_SCALE}" \
  --undetermined_v3_selector_unique_bonus_scale "${V3_SELECTOR_UNIQUE_BONUS_SCALE}" \
  --undetermined_v3_selector_pending_penalty_scale "${V3_SELECTOR_PENDING_PENALTY_SCALE}" \
  --undetermined_v3_selector_duplicate_penalty_scale "${V3_SELECTOR_DUPLICATE_PENALTY_SCALE}" \
  --undetermined_v3_selector_progress_bonus_scale "${V3_SELECTOR_PROGRESS_BONUS_SCALE}" \
  --undetermined_v2_sl_dense_scale "${UNDET_V2_SL_DENSE_SCALE}" \
  --undetermined_v2_sl_delta_scale "${UNDET_V2_SL_DELTA_SCALE}" \
  --undetermined_v2_sl_success_scale "${UNDET_V2_SL_SUCCESS_SCALE}" \
  --undetermined_v2_sl_success_threshold "${UNDET_V2_SL_SUCCESS_THRESHOLD}" \
  --nd_arrival_reward "${ND_ARRIVAL_REWARD}" \
  --nd_goal_terminal_reward "${ND_GOAL_TERMINAL_REWARD}" \
  --undetermined_v3_target_kl_coef 0.02 \
  --entropy_coef "${ENTROPY_COEF}" \
  --undetermined_v3_curriculum_motion_phase_ratio "${V3_CURRICULUM_MOTION_PHASE_RATIO}" \
  --undetermined_v3_curriculum_motion_reward_boost "${V3_CURRICULUM_MOTION_REWARD_BOOST}" \
  --undetermined_v3_curriculum_selector_reward_scale_early "${V3_CURRICULUM_SELECTOR_REWARD_EARLY}" \
  --undetermined_v3_curriculum_selector_reward_scale_late "${V3_CURRICULUM_SELECTOR_REWARD_LATE}" \
  --undetermined_v3_curriculum_selector_kl_scale_early "${V3_CURRICULUM_SELECTOR_KL_EARLY}" \
  --undetermined_v3_curriculum_selector_kl_scale_late "${V3_CURRICULUM_SELECTOR_KL_LATE}" \
  --undet_v3_target_latent_model_dir "${LATENT_MODEL_DIR}" \
  --undet_v3_latent_train_mode "${LATENT_TRAIN_MODE}" \
  --num_mini_batch "${NUM_MINI_BATCH}" \
  "$@"

echo "Finished. Log saved to: ${LOG_FILE}"
