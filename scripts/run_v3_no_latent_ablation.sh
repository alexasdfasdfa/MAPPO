#!/bin/bash
set -euo pipefail
echo "no latent ablation script placeholder"
#!/bin/bash
# Submit from MAPPO root:
#   cd /home/wangdx_lab/cse12211818/MAPPO && sbatch scripts/run_v3_no_latent_ablation.sh
#
# Ablation:
#   v3 MAPPO training WITHOUT pretrained latent selector checkpoint.

#SBATCH -o swarm.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J mappo-v3-no-latent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

date
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate swE2
else
  echo "[warn] conda not in PATH; use env with python+torch" >&2
fi

ROBOT_INITIAL_SPAWN_MODE="${ROBOT_INITIAL_SPAWN_MODE:-random_box}"
V3_DISABLE_ATTN_TAIL="${V3_DISABLE_ATTN_TAIL:-1}"
V3_DISABLE_PREV_TARGET="${V3_DISABLE_PREV_TARGET:-1}"
RANDOMIZE_INITIAL_POSITIONS="${RANDOMIZE_INITIAL_POSITIONS:-1}"
V3_REWARD_ENABLE="${V3_REWARD_ENABLE:-1}"
V3_REWARD_M_DROP_SCALE="${V3_REWARD_M_DROP_SCALE:-0.60}"
V3_REWARD_S_DROP_SCALE="${V3_REWARD_S_DROP_SCALE:-0.15}"
V3_REWARD_TRAVEL_PENALTY_SCALE="${V3_REWARD_TRAVEL_PENALTY_SCALE:-0.02}"
V3_SELECTOR_UNIQUE_BONUS_SCALE="${V3_SELECTOR_UNIQUE_BONUS_SCALE:-0.10}"
V3_SELECTOR_PENDING_PENALTY_SCALE="${V3_SELECTOR_PENDING_PENALTY_SCALE:-0.20}"
V3_SELECTOR_DUPLICATE_PENALTY_SCALE="${V3_SELECTOR_DUPLICATE_PENALTY_SCALE:-0.20}"
V3_SELECTOR_PROGRESS_BONUS_SCALE="${V3_SELECTOR_PROGRESS_BONUS_SCALE:-0.20}"
UNDET_V2_SL_DENSE_SCALE="${UNDET_V2_SL_DENSE_SCALE:-0.20}"
UNDET_V2_SL_DELTA_SCALE="${UNDET_V2_SL_DELTA_SCALE:-0.60}"
UNDET_V2_SL_SUCCESS_SCALE="${UNDET_V2_SL_SUCCESS_SCALE:-2.00}"
UNDET_V2_SL_SUCCESS_THRESHOLD="${UNDET_V2_SL_SUCCESS_THRESHOLD:-0.97}"
ND_ARRIVAL_REWARD="${ND_ARRIVAL_REWARD:-0.60}"
ND_GOAL_TERMINAL_REWARD="${ND_GOAL_TERMINAL_REWARD:-1.20}"
UNDET_V3_TARGET_KL_COEF="${UNDET_V3_TARGET_KL_COEF:-0.02}"
ENTROPY_COEF="${ENTROPY_COEF:-0.02}"
V3_CURRICULUM_ENABLE="${V3_CURRICULUM_ENABLE:-1}"
V3_CURRICULUM_MOTION_PHASE_RATIO="${V3_CURRICULUM_MOTION_PHASE_RATIO:-0.45}"
V3_CURRICULUM_MOTION_REWARD_BOOST="${V3_CURRICULUM_MOTION_REWARD_BOOST:-1.6}"
V3_CURRICULUM_SELECTOR_REWARD_EARLY="${V3_CURRICULUM_SELECTOR_REWARD_EARLY:-0.35}"
V3_CURRICULUM_SELECTOR_REWARD_LATE="${V3_CURRICULUM_SELECTOR_REWARD_LATE:-1.35}"
V3_CURRICULUM_SELECTOR_KL_EARLY="${V3_CURRICULUM_SELECTOR_KL_EARLY:-0.25}"
V3_CURRICULUM_SELECTOR_KL_LATE="${V3_CURRICULUM_SELECTOR_KL_LATE:-1.50}"

# Target-swap mode switch (v3):
TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-0}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi

V3_DISABLE_ATTN_TAIL_FLAG=""
if [[ "${V3_DISABLE_ATTN_TAIL}" == "1" ]]; then
  V3_DISABLE_ATTN_TAIL_FLAG="--undetermined_v3_disable_attn_tail_for_motion"
fi

V3_DISABLE_PREV_TARGET_FLAG=""
if [[ "${V3_DISABLE_PREV_TARGET}" == "1" ]]; then
  V3_DISABLE_PREV_TARGET_FLAG="--undetermined_v3_disable_prev_target_for_motion"
fi

RANDOMIZE_INITIAL_POSITIONS_FLAG=""
if [[ "${RANDOMIZE_INITIAL_POSITIONS}" == "1" ]]; then
  RANDOMIZE_INITIAL_POSITIONS_FLAG="--randomize_robot_initial_positions"
fi

V3_REWARD_ENABLE_FLAG=""
if [[ "${V3_REWARD_ENABLE}" == "1" ]]; then
  V3_REWARD_ENABLE_FLAG="--undetermined_v3_reward_enable"
fi

V3_CURRICULUM_ENABLE_FLAG=""
if [[ "${V3_CURRICULUM_ENABLE}" == "1" ]]; then
  V3_CURRICULUM_ENABLE_FLAG="--undetermined_v3_curriculum_enable"
fi

echo "ROOT=${ROOT}"
echo "ABLATION_MODE=no_pretrained_latent"
echo "ROBOT_INITIAL_SPAWN_MODE=${ROBOT_INITIAL_SPAWN_MODE}"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "V3_DISABLE_ATTN_TAIL=${V3_DISABLE_ATTN_TAIL}"
echo "V3_DISABLE_PREV_TARGET=${V3_DISABLE_PREV_TARGET}"
echo "RANDOMIZE_INITIAL_POSITIONS=${RANDOMIZE_INITIAL_POSITIONS}"
echo "V3_REWARD_ENABLE=${V3_REWARD_ENABLE}"
echo "V3_REWARD_M_DROP_SCALE=${V3_REWARD_M_DROP_SCALE}"
echo "V3_REWARD_S_DROP_SCALE=${V3_REWARD_S_DROP_SCALE}"
echo "V3_REWARD_TRAVEL_PENALTY_SCALE=${V3_REWARD_TRAVEL_PENALTY_SCALE}"
echo "V3_SELECTOR_UNIQUE_BONUS_SCALE=${V3_SELECTOR_UNIQUE_BONUS_SCALE}"
echo "V3_SELECTOR_PENDING_PENALTY_SCALE=${V3_SELECTOR_PENDING_PENALTY_SCALE}"
echo "V3_SELECTOR_DUPLICATE_PENALTY_SCALE=${V3_SELECTOR_DUPLICATE_PENALTY_SCALE}"
echo "V3_SELECTOR_PROGRESS_BONUS_SCALE=${V3_SELECTOR_PROGRESS_BONUS_SCALE}"
echo "UNDET_V2_SL_DENSE_SCALE=${UNDET_V2_SL_DENSE_SCALE}"
echo "UNDET_V2_SL_DELTA_SCALE=${UNDET_V2_SL_DELTA_SCALE}"
echo "UNDET_V2_SL_SUCCESS_SCALE=${UNDET_V2_SL_SUCCESS_SCALE}"
echo "UNDET_V2_SL_SUCCESS_THRESHOLD=${UNDET_V2_SL_SUCCESS_THRESHOLD}"
echo "ND_ARRIVAL_REWARD=${ND_ARRIVAL_REWARD}"
echo "ND_GOAL_TERMINAL_REWARD=${ND_GOAL_TERMINAL_REWARD}"
echo "UNDET_V3_TARGET_KL_COEF=${UNDET_V3_TARGET_KL_COEF}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "V3_CURRICULUM_ENABLE=${V3_CURRICULUM_ENABLE}"
echo "V3_CURRICULUM_MOTION_PHASE_RATIO=${V3_CURRICULUM_MOTION_PHASE_RATIO}"
echo "V3_CURRICULUM_MOTION_REWARD_BOOST=${V3_CURRICULUM_MOTION_REWARD_BOOST}"
echo "V3_CURRICULUM_SELECTOR_REWARD_EARLY=${V3_CURRICULUM_SELECTOR_REWARD_EARLY}"
echo "V3_CURRICULUM_SELECTOR_REWARD_LATE=${V3_CURRICULUM_SELECTOR_REWARD_LATE}"
echo "V3_CURRICULUM_SELECTOR_KL_EARLY=${V3_CURRICULUM_SELECTOR_KL_EARLY}"
echo "V3_CURRICULUM_SELECTOR_KL_LATE=${V3_CURRICULUM_SELECTOR_KL_LATE}"
echo "Python=$(command -v python)"

python train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
  ${V3_DISABLE_ATTN_TAIL_FLAG} \
  ${V3_DISABLE_PREV_TARGET_FLAG} \
  ${RANDOMIZE_INITIAL_POSITIONS_FLAG} \
  ${V3_REWARD_ENABLE_FLAG} \
  ${V3_CURRICULUM_ENABLE_FLAG} \
  --robot_initial_spawn_mode "${ROBOT_INITIAL_SPAWN_MODE}" \
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
  --undetermined_v3_target_kl_coef "${UNDET_V3_TARGET_KL_COEF}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --undetermined_v3_curriculum_motion_phase_ratio "${V3_CURRICULUM_MOTION_PHASE_RATIO}" \
  --undetermined_v3_curriculum_motion_reward_boost "${V3_CURRICULUM_MOTION_REWARD_BOOST}" \
  --undetermined_v3_curriculum_selector_reward_scale_early "${V3_CURRICULUM_SELECTOR_REWARD_EARLY}" \
  --undetermined_v3_curriculum_selector_reward_scale_late "${V3_CURRICULUM_SELECTOR_REWARD_LATE}" \
  --undetermined_v3_curriculum_selector_kl_scale_early "${V3_CURRICULUM_SELECTOR_KL_EARLY}" \
  --undetermined_v3_curriculum_selector_kl_scale_late "${V3_CURRICULUM_SELECTOR_KL_LATE}" \
  "$@"

date
#!/bin/bash
# Submit from MAPPO root:
#   cd /home/wangdx_lab/cse12211818/MAPPO && sbatch scripts/run_v3_no_latent_ablation.sh
#
# Pipeline:
#   MAPPO v3 selector + motion policy training from scratch (no pretrained latent checkpoint)

#SBATCH -o swarm.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J mappo-v3-no-latent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

date
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate swE2
else
  echo "[warn] conda not in PATH; use env with python+torch" >&2
fi

ROBOT_INITIAL_SPAWN_MODE="${ROBOT_INITIAL_SPAWN_MODE:-random_box}"
V3_DISABLE_ATTN_TAIL="${V3_DISABLE_ATTN_TAIL:-1}"
V3_DISABLE_PREV_TARGET="${V3_DISABLE_PREV_TARGET:-1}"
RANDOMIZE_INITIAL_POSITIONS="${RANDOMIZE_INITIAL_POSITIONS:-1}"
V3_REWARD_ENABLE="${V3_REWARD_ENABLE:-1}"
V3_REWARD_M_DROP_SCALE="${V3_REWARD_M_DROP_SCALE:-0.60}"
V3_REWARD_S_DROP_SCALE="${V3_REWARD_S_DROP_SCALE:-0.15}"
V3_REWARD_TRAVEL_PENALTY_SCALE="${V3_REWARD_TRAVEL_PENALTY_SCALE:-0.02}"
V3_SELECTOR_UNIQUE_BONUS_SCALE="${V3_SELECTOR_UNIQUE_BONUS_SCALE:-0.10}"
V3_SELECTOR_PENDING_PENALTY_SCALE="${V3_SELECTOR_PENDING_PENALTY_SCALE:-0.20}"
V3_SELECTOR_DUPLICATE_PENALTY_SCALE="${V3_SELECTOR_DUPLICATE_PENALTY_SCALE:-0.20}"
V3_SELECTOR_PROGRESS_BONUS_SCALE="${V3_SELECTOR_PROGRESS_BONUS_SCALE:-0.20}"
UNDET_V2_SL_DENSE_SCALE="${UNDET_V2_SL_DENSE_SCALE:-0.20}"
UNDET_V2_SL_DELTA_SCALE="${UNDET_V2_SL_DELTA_SCALE:-0.60}"
UNDET_V2_SL_SUCCESS_SCALE="${UNDET_V2_SL_SUCCESS_SCALE:-2.00}"
UNDET_V2_SL_SUCCESS_THRESHOLD="${UNDET_V2_SL_SUCCESS_THRESHOLD:-0.97}"
ND_ARRIVAL_REWARD="${ND_ARRIVAL_REWARD:-0.60}"
ND_GOAL_TERMINAL_REWARD="${ND_GOAL_TERMINAL_REWARD:-1.20}"
UNDET_V3_TARGET_KL_COEF="${UNDET_V3_TARGET_KL_COEF:-0.02}"
ENTROPY_COEF="${ENTROPY_COEF:-0.02}"
V3_CURRICULUM_ENABLE="${V3_CURRICULUM_ENABLE:-1}"
V3_CURRICULUM_MOTION_PHASE_RATIO="${V3_CURRICULUM_MOTION_PHASE_RATIO:-0.45}"
V3_CURRICULUM_MOTION_REWARD_BOOST="${V3_CURRICULUM_MOTION_REWARD_BOOST:-1.6}"
V3_CURRICULUM_SELECTOR_REWARD_EARLY="${V3_CURRICULUM_SELECTOR_REWARD_EARLY:-0.35}"
V3_CURRICULUM_SELECTOR_REWARD_LATE="${V3_CURRICULUM_SELECTOR_REWARD_LATE:-1.35}"
V3_CURRICULUM_SELECTOR_KL_EARLY="${V3_CURRICULUM_SELECTOR_KL_EARLY:-0.25}"
V3_CURRICULUM_SELECTOR_KL_LATE="${V3_CURRICULUM_SELECTOR_KL_LATE:-1.50}"
# Target-swap mode switch (v3):
TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-0}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi
V3_DISABLE_ATTN_TAIL_FLAG=""
if [[ "${V3_DISABLE_ATTN_TAIL}" == "1" ]]; then
  V3_DISABLE_ATTN_TAIL_FLAG="--undetermined_v3_disable_attn_tail_for_motion"
fi
V3_DISABLE_PREV_TARGET_FLAG=""
if [[ "${V3_DISABLE_PREV_TARGET}" == "1" ]]; then
  V3_DISABLE_PREV_TARGET_FLAG="--undetermined_v3_disable_prev_target_for_motion"
fi
RANDOMIZE_INITIAL_POSITIONS_FLAG=""
if [[ "${RANDOMIZE_INITIAL_POSITIONS}" == "1" ]]; then
  RANDOMIZE_INITIAL_POSITIONS_FLAG="--randomize_robot_initial_positions"
fi
V3_REWARD_ENABLE_FLAG=""
if [[ "${V3_REWARD_ENABLE}" == "1" ]]; then
  V3_REWARD_ENABLE_FLAG="--undetermined_v3_reward_enable"
fi
V3_CURRICULUM_ENABLE_FLAG=""
if [[ "${V3_CURRICULUM_ENABLE}" == "1" ]]; then
  V3_CURRICULUM_ENABLE_FLAG="--undetermined_v3_curriculum_enable"
fi

echo "ROOT=${ROOT}"
echo "MODE=no_latent_ablation"
echo "ROBOT_INITIAL_SPAWN_MODE=${ROBOT_INITIAL_SPAWN_MODE}"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "V3_DISABLE_ATTN_TAIL=${V3_DISABLE_ATTN_TAIL}"
echo "V3_DISABLE_PREV_TARGET=${V3_DISABLE_PREV_TARGET}"
echo "RANDOMIZE_INITIAL_POSITIONS=${RANDOMIZE_INITIAL_POSITIONS}"
echo "V3_REWARD_ENABLE=${V3_REWARD_ENABLE}"
echo "V3_REWARD_M_DROP_SCALE=${V3_REWARD_M_DROP_SCALE}"
echo "V3_REWARD_S_DROP_SCALE=${V3_REWARD_S_DROP_SCALE}"
echo "V3_REWARD_TRAVEL_PENALTY_SCALE=${V3_REWARD_TRAVEL_PENALTY_SCALE}"
echo "V3_SELECTOR_UNIQUE_BONUS_SCALE=${V3_SELECTOR_UNIQUE_BONUS_SCALE}"
echo "V3_SELECTOR_PENDING_PENALTY_SCALE=${V3_SELECTOR_PENDING_PENALTY_SCALE}"
echo "V3_SELECTOR_DUPLICATE_PENALTY_SCALE=${V3_SELECTOR_DUPLICATE_PENALTY_SCALE}"
echo "V3_SELECTOR_PROGRESS_BONUS_SCALE=${V3_SELECTOR_PROGRESS_BONUS_SCALE}"
echo "UNDET_V2_SL_DENSE_SCALE=${UNDET_V2_SL_DENSE_SCALE}"
echo "UNDET_V2_SL_DELTA_SCALE=${UNDET_V2_SL_DELTA_SCALE}"
echo "UNDET_V2_SL_SUCCESS_SCALE=${UNDET_V2_SL_SUCCESS_SCALE}"
echo "UNDET_V2_SL_SUCCESS_THRESHOLD=${UNDET_V2_SL_SUCCESS_THRESHOLD}"
echo "ND_ARRIVAL_REWARD=${ND_ARRIVAL_REWARD}"
echo "ND_GOAL_TERMINAL_REWARD=${ND_GOAL_TERMINAL_REWARD}"
echo "UNDET_V3_TARGET_KL_COEF=${UNDET_V3_TARGET_KL_COEF}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "V3_CURRICULUM_ENABLE=${V3_CURRICULUM_ENABLE}"
echo "V3_CURRICULUM_MOTION_PHASE_RATIO=${V3_CURRICULUM_MOTION_PHASE_RATIO}"
echo "V3_CURRICULUM_MOTION_REWARD_BOOST=${V3_CURRICULUM_MOTION_REWARD_BOOST}"
echo "V3_CURRICULUM_SELECTOR_REWARD_EARLY=${V3_CURRICULUM_SELECTOR_REWARD_EARLY}"
echo "V3_CURRICULUM_SELECTOR_REWARD_LATE=${V3_CURRICULUM_SELECTOR_REWARD_LATE}"
echo "V3_CURRICULUM_SELECTOR_KL_EARLY=${V3_CURRICULUM_SELECTOR_KL_EARLY}"
echo "V3_CURRICULUM_SELECTOR_KL_LATE=${V3_CURRICULUM_SELECTOR_KL_LATE}"
echo "Python=$(command -v python)"

python train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
  ${V3_DISABLE_ATTN_TAIL_FLAG} \
  ${V3_DISABLE_PREV_TARGET_FLAG} \
  ${RANDOMIZE_INITIAL_POSITIONS_FLAG} \
  ${V3_REWARD_ENABLE_FLAG} \
  ${V3_CURRICULUM_ENABLE_FLAG} \
  --robot_initial_spawn_mode "${ROBOT_INITIAL_SPAWN_MODE}" \
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
  --undetermined_v3_target_kl_coef "${UNDET_V3_TARGET_KL_COEF}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --undetermined_v3_curriculum_motion_phase_ratio "${V3_CURRICULUM_MOTION_PHASE_RATIO}" \
  --undetermined_v3_curriculum_motion_reward_boost "${V3_CURRICULUM_MOTION_REWARD_BOOST}" \
  --undetermined_v3_curriculum_selector_reward_scale_early "${V3_CURRICULUM_SELECTOR_REWARD_EARLY}" \
  --undetermined_v3_curriculum_selector_reward_scale_late "${V3_CURRICULUM_SELECTOR_REWARD_LATE}" \
  --undetermined_v3_curriculum_selector_kl_scale_early "${V3_CURRICULUM_SELECTOR_KL_EARLY}" \
  --undetermined_v3_curriculum_selector_kl_scale_late "${V3_CURRICULUM_SELECTOR_KL_LATE}" \
  "$@"

date
#!/bin/bash
# Submit from MAPPO root:
#   cd /home/wangdx_lab/cse12211818/MAPPO && sbatch scripts/run_v3_no_latent_ablation.sh
#
# Ablation:
#   v3 MAPPO training WITHOUT pretrained latent selector checkpoint.

#SBATCH -o swarm.%j.out
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH -J mappo-v3-no-latent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

date
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate swE2
else
  echo "[warn] conda not in PATH; use env with python+torch" >&2
fi

ROBOT_INITIAL_SPAWN_MODE="${ROBOT_INITIAL_SPAWN_MODE:-random_box}"
V3_DISABLE_ATTN_TAIL="${V3_DISABLE_ATTN_TAIL:-1}"
V3_DISABLE_PREV_TARGET="${V3_DISABLE_PREV_TARGET:-1}"
RANDOMIZE_INITIAL_POSITIONS="${RANDOMIZE_INITIAL_POSITIONS:-1}"
V3_REWARD_ENABLE="${V3_REWARD_ENABLE:-1}"
V3_REWARD_M_DROP_SCALE="${V3_REWARD_M_DROP_SCALE:-0.60}"
V3_REWARD_S_DROP_SCALE="${V3_REWARD_S_DROP_SCALE:-0.15}"
V3_REWARD_TRAVEL_PENALTY_SCALE="${V3_REWARD_TRAVEL_PENALTY_SCALE:-0.02}"
V3_SELECTOR_UNIQUE_BONUS_SCALE="${V3_SELECTOR_UNIQUE_BONUS_SCALE:-0.10}"
V3_SELECTOR_PENDING_PENALTY_SCALE="${V3_SELECTOR_PENDING_PENALTY_SCALE:-0.20}"
V3_SELECTOR_DUPLICATE_PENALTY_SCALE="${V3_SELECTOR_DUPLICATE_PENALTY_SCALE:-0.20}"
V3_SELECTOR_PROGRESS_BONUS_SCALE="${V3_SELECTOR_PROGRESS_BONUS_SCALE:-0.20}"
UNDET_V2_SL_DENSE_SCALE="${UNDET_V2_SL_DENSE_SCALE:-0.20}"
UNDET_V2_SL_DELTA_SCALE="${UNDET_V2_SL_DELTA_SCALE:-0.60}"
UNDET_V2_SL_SUCCESS_SCALE="${UNDET_V2_SL_SUCCESS_SCALE:-2.00}"
UNDET_V2_SL_SUCCESS_THRESHOLD="${UNDET_V2_SL_SUCCESS_THRESHOLD:-0.97}"
ND_ARRIVAL_REWARD="${ND_ARRIVAL_REWARD:-0.60}"
ND_GOAL_TERMINAL_REWARD="${ND_GOAL_TERMINAL_REWARD:-1.20}"
UNDET_V3_TARGET_KL_COEF="${UNDET_V3_TARGET_KL_COEF:-0.02}"
ENTROPY_COEF="${ENTROPY_COEF:-0.02}"

# Target-swap mode switch (v3):
# - TARGET_EXCHANGE_ENABLE=1 (default): enable mutual target swap policy.
# - TARGET_EXCHANGE_ENABLE=0         : disable target swap and run without exchange.
TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-0}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi

V3_DISABLE_ATTN_TAIL_FLAG=""
if [[ "${V3_DISABLE_ATTN_TAIL}" == "1" ]]; then
  V3_DISABLE_ATTN_TAIL_FLAG="--undetermined_v3_disable_attn_tail_for_motion"
fi

V3_DISABLE_PREV_TARGET_FLAG=""
if [[ "${V3_DISABLE_PREV_TARGET}" == "1" ]]; then
  V3_DISABLE_PREV_TARGET_FLAG="--undetermined_v3_disable_prev_target_for_motion"
fi

RANDOMIZE_INITIAL_POSITIONS_FLAG=""
if [[ "${RANDOMIZE_INITIAL_POSITIONS}" == "1" ]]; then
  RANDOMIZE_INITIAL_POSITIONS_FLAG="--randomize_robot_initial_positions"
fi

V3_REWARD_ENABLE_FLAG=""
if [[ "${V3_REWARD_ENABLE}" == "1" ]]; then
  V3_REWARD_ENABLE_FLAG="--undetermined_v3_reward_enable"
fi

echo "ROOT=${ROOT}"
echo "ABLATION_MODE=no_pretrained_latent"
echo "ROBOT_INITIAL_SPAWN_MODE=${ROBOT_INITIAL_SPAWN_MODE}"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "V3_DISABLE_ATTN_TAIL=${V3_DISABLE_ATTN_TAIL}"
echo "V3_DISABLE_PREV_TARGET=${V3_DISABLE_PREV_TARGET}"
echo "RANDOMIZE_INITIAL_POSITIONS=${RANDOMIZE_INITIAL_POSITIONS}"
echo "V3_REWARD_ENABLE=${V3_REWARD_ENABLE}"
echo "V3_REWARD_M_DROP_SCALE=${V3_REWARD_M_DROP_SCALE}"
echo "V3_REWARD_S_DROP_SCALE=${V3_REWARD_S_DROP_SCALE}"
echo "V3_REWARD_TRAVEL_PENALTY_SCALE=${V3_REWARD_TRAVEL_PENALTY_SCALE}"
echo "V3_SELECTOR_UNIQUE_BONUS_SCALE=${V3_SELECTOR_UNIQUE_BONUS_SCALE}"
echo "V3_SELECTOR_PENDING_PENALTY_SCALE=${V3_SELECTOR_PENDING_PENALTY_SCALE}"
echo "V3_SELECTOR_DUPLICATE_PENALTY_SCALE=${V3_SELECTOR_DUPLICATE_PENALTY_SCALE}"
echo "V3_SELECTOR_PROGRESS_BONUS_SCALE=${V3_SELECTOR_PROGRESS_BONUS_SCALE}"
echo "UNDET_V2_SL_DENSE_SCALE=${UNDET_V2_SL_DENSE_SCALE}"
echo "UNDET_V2_SL_DELTA_SCALE=${UNDET_V2_SL_DELTA_SCALE}"
echo "UNDET_V2_SL_SUCCESS_SCALE=${UNDET_V2_SL_SUCCESS_SCALE}"
echo "UNDET_V2_SL_SUCCESS_THRESHOLD=${UNDET_V2_SL_SUCCESS_THRESHOLD}"
echo "ND_ARRIVAL_REWARD=${ND_ARRIVAL_REWARD}"
echo "ND_GOAL_TERMINAL_REWARD=${ND_GOAL_TERMINAL_REWARD}"
echo "UNDET_V3_TARGET_KL_COEF=${UNDET_V3_TARGET_KL_COEF}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "Python=$(command -v python)"

python train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
  ${V3_DISABLE_ATTN_TAIL_FLAG} \
  ${V3_DISABLE_PREV_TARGET_FLAG} \
  ${RANDOMIZE_INITIAL_POSITIONS_FLAG} \
  ${V3_REWARD_ENABLE_FLAG} \
  --robot_initial_spawn_mode "${ROBOT_INITIAL_SPAWN_MODE}" \
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
  --undetermined_v3_target_kl_coef "${UNDET_V3_TARGET_KL_COEF}" \
  --entropy_coef "${ENTROPY_COEF}" \
  "$@"

date
#!/bin/bash
# Submit from MAPPO root:
#   cd /home/wangdx_lab/cse12211818/MAPPO && sbatch scripts/run_v3_no_latent_ablation.sh
#
# Ablation:
#   v3 MAPPO training WITHOUT pretrained latent selector checkpoint.

#SBATCH -o swarm.%j.out
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH -J mappo-v3-no-latent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

date
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate swE2
else
  echo "[warn] conda not in PATH; use env with python+torch" >&2
fi

ROBOT_INITIAL_SPAWN_MODE="${ROBOT_INITIAL_SPAWN_MODE:-random_box}"
V3_DISABLE_ATTN_TAIL="${V3_DISABLE_ATTN_TAIL:-1}"
V3_DISABLE_PREV_TARGET="${V3_DISABLE_PREV_TARGET:-1}"
RANDOMIZE_INITIAL_POSITIONS="${RANDOMIZE_INITIAL_POSITIONS:-1}"
V3_REWARD_ENABLE="${V3_REWARD_ENABLE:-1}"
V3_REWARD_M_DROP_SCALE="${V3_REWARD_M_DROP_SCALE:-0.60}"
V3_REWARD_S_DROP_SCALE="${V3_REWARD_S_DROP_SCALE:-0.15}"
V3_REWARD_TRAVEL_PENALTY_SCALE="${V3_REWARD_TRAVEL_PENALTY_SCALE:-0.02}"
V3_SELECTOR_UNIQUE_BONUS_SCALE="${V3_SELECTOR_UNIQUE_BONUS_SCALE:-0.10}"
V3_SELECTOR_PENDING_PENALTY_SCALE="${V3_SELECTOR_PENDING_PENALTY_SCALE:-0.20}"
V3_SELECTOR_DUPLICATE_PENALTY_SCALE="${V3_SELECTOR_DUPLICATE_PENALTY_SCALE:-0.20}"
V3_SELECTOR_PROGRESS_BONUS_SCALE="${V3_SELECTOR_PROGRESS_BONUS_SCALE:-0.20}"
UNDET_V2_SL_DENSE_SCALE="${UNDET_V2_SL_DENSE_SCALE:-0.20}"
UNDET_V2_SL_DELTA_SCALE="${UNDET_V2_SL_DELTA_SCALE:-0.60}"
UNDET_V2_SL_SUCCESS_SCALE="${UNDET_V2_SL_SUCCESS_SCALE:-2.00}"
UNDET_V2_SL_SUCCESS_THRESHOLD="${UNDET_V2_SL_SUCCESS_THRESHOLD:-0.97}"
ND_ARRIVAL_REWARD="${ND_ARRIVAL_REWARD:-0.60}"
ND_GOAL_TERMINAL_REWARD="${ND_GOAL_TERMINAL_REWARD:-1.20}"
UNDET_V3_TARGET_KL_COEF="${UNDET_V3_TARGET_KL_COEF:-0.02}"
ENTROPY_COEF="${ENTROPY_COEF:-0.02}"

# Target-swap mode switch (v3):
# - TARGET_EXCHANGE_ENABLE=1 (default): enable mutual target swap policy.
# - TARGET_EXCHANGE_ENABLE=0         : disable target swap and run without exchange.
TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-0}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi

V3_DISABLE_ATTN_TAIL_FLAG=""
if [[ "${V3_DISABLE_ATTN_TAIL}" == "1" ]]; then
  V3_DISABLE_ATTN_TAIL_FLAG="--undetermined_v3_disable_attn_tail_for_motion"
fi

V3_DISABLE_PREV_TARGET_FLAG=""
if [[ "${V3_DISABLE_PREV_TARGET}" == "1" ]]; then
  V3_DISABLE_PREV_TARGET_FLAG="--undetermined_v3_disable_prev_target_for_motion"
fi

RANDOMIZE_INITIAL_POSITIONS_FLAG=""
if [[ "${RANDOMIZE_INITIAL_POSITIONS}" == "1" ]]; then
  RANDOMIZE_INITIAL_POSITIONS_FLAG="--randomize_robot_initial_positions"
fi

V3_REWARD_ENABLE_FLAG=""
if [[ "${V3_REWARD_ENABLE}" == "1" ]]; then
  V3_REWARD_ENABLE_FLAG="--undetermined_v3_reward_enable"
fi

echo "ROOT=${ROOT}"
echo "ABLATION_MODE=no_pretrained_latent"
echo "ROBOT_INITIAL_SPAWN_MODE=${ROBOT_INITIAL_SPAWN_MODE}"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "V3_DISABLE_ATTN_TAIL=${V3_DISABLE_ATTN_TAIL}"
echo "V3_DISABLE_PREV_TARGET=${V3_DISABLE_PREV_TARGET}"
echo "RANDOMIZE_INITIAL_POSITIONS=${RANDOMIZE_INITIAL_POSITIONS}"
echo "V3_REWARD_ENABLE=${V3_REWARD_ENABLE}"
echo "V3_REWARD_M_DROP_SCALE=${V3_REWARD_M_DROP_SCALE}"
echo "V3_REWARD_S_DROP_SCALE=${V3_REWARD_S_DROP_SCALE}"
echo "V3_REWARD_TRAVEL_PENALTY_SCALE=${V3_REWARD_TRAVEL_PENALTY_SCALE}"
echo "V3_SELECTOR_UNIQUE_BONUS_SCALE=${V3_SELECTOR_UNIQUE_BONUS_SCALE}"
echo "V3_SELECTOR_PENDING_PENALTY_SCALE=${V3_SELECTOR_PENDING_PENALTY_SCALE}"
echo "V3_SELECTOR_DUPLICATE_PENALTY_SCALE=${V3_SELECTOR_DUPLICATE_PENALTY_SCALE}"
echo "V3_SELECTOR_PROGRESS_BONUS_SCALE=${V3_SELECTOR_PROGRESS_BONUS_SCALE}"
echo "UNDET_V2_SL_DENSE_SCALE=${UNDET_V2_SL_DENSE_SCALE}"
echo "UNDET_V2_SL_DELTA_SCALE=${UNDET_V2_SL_DELTA_SCALE}"
echo "UNDET_V2_SL_SUCCESS_SCALE=${UNDET_V2_SL_SUCCESS_SCALE}"
echo "UNDET_V2_SL_SUCCESS_THRESHOLD=${UNDET_V2_SL_SUCCESS_THRESHOLD}"
echo "ND_ARRIVAL_REWARD=${ND_ARRIVAL_REWARD}"
echo "ND_GOAL_TERMINAL_REWARD=${ND_GOAL_TERMINAL_REWARD}"
echo "UNDET_V3_TARGET_KL_COEF=${UNDET_V3_TARGET_KL_COEF}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "Python=$(command -v python)"

python train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
  ${V3_DISABLE_ATTN_TAIL_FLAG} \
  ${V3_DISABLE_PREV_TARGET_FLAG} \
  ${RANDOMIZE_INITIAL_POSITIONS_FLAG} \
  ${V3_REWARD_ENABLE_FLAG} \
  --robot_initial_spawn_mode "${ROBOT_INITIAL_SPAWN_MODE}" \
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
  --undetermined_v3_target_kl_coef "${UNDET_V3_TARGET_KL_COEF}" \
  --entropy_coef "${ENTROPY_COEF}" \
  "$@"

date
#!/bin/bash
# Submit from MAPPO root:
#   cd /home/wangdx_lab/cse12211818/MAPPO && sbatch scripts/run_v3_no_latent_ablation.sh
#
# Ablation:
#   v3 MAPPO training WITHOUT pretrained latent selector checkpoint.

#SBATCH -o swarm.%j.out
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH -J mappo-v3-no-latent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

date
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate swE2
else
  echo "[warn] conda not in PATH; use env with python+torch" >&2
fi

ROBOT_INITIAL_SPAWN_MODE="${ROBOT_INITIAL_SPAWN_MODE:-random_box}"
V3_DISABLE_ATTN_TAIL="${V3_DISABLE_ATTN_TAIL:-1}"
V3_DISABLE_PREV_TARGET="${V3_DISABLE_PREV_TARGET:-1}"
RANDOMIZE_INITIAL_POSITIONS="${RANDOMIZE_INITIAL_POSITIONS:-1}"
V3_REWARD_ENABLE="${V3_REWARD_ENABLE:-1}"
V3_REWARD_M_DROP_SCALE="${V3_REWARD_M_DROP_SCALE:-0.60}"
V3_REWARD_S_DROP_SCALE="${V3_REWARD_S_DROP_SCALE:-0.15}"
V3_REWARD_TRAVEL_PENALTY_SCALE="${V3_REWARD_TRAVEL_PENALTY_SCALE:-0.02}"
V3_SELECTOR_UNIQUE_BONUS_SCALE="${V3_SELECTOR_UNIQUE_BONUS_SCALE:-0.10}"
V3_SELECTOR_PENDING_PENALTY_SCALE="${V3_SELECTOR_PENDING_PENALTY_SCALE:-0.20}"
V3_SELECTOR_DUPLICATE_PENALTY_SCALE="${V3_SELECTOR_DUPLICATE_PENALTY_SCALE:-0.20}"
V3_SELECTOR_PROGRESS_BONUS_SCALE="${V3_SELECTOR_PROGRESS_BONUS_SCALE:-0.20}"
UNDET_V2_SL_DENSE_SCALE="${UNDET_V2_SL_DENSE_SCALE:-0.20}"
UNDET_V2_SL_DELTA_SCALE="${UNDET_V2_SL_DELTA_SCALE:-0.60}"
UNDET_V2_SL_SUCCESS_SCALE="${UNDET_V2_SL_SUCCESS_SCALE:-2.00}"
UNDET_V2_SL_SUCCESS_THRESHOLD="${UNDET_V2_SL_SUCCESS_THRESHOLD:-0.97}"
ND_ARRIVAL_REWARD="${ND_ARRIVAL_REWARD:-0.60}"
ND_GOAL_TERMINAL_REWARD="${ND_GOAL_TERMINAL_REWARD:-1.20}"
UNDET_V3_TARGET_KL_COEF="${UNDET_V3_TARGET_KL_COEF:-0.02}"
ENTROPY_COEF="${ENTROPY_COEF:-0.02}"

# Target-swap mode switch (v3):
# - TARGET_EXCHANGE_ENABLE=1 (default): enable mutual target swap policy.
# - TARGET_EXCHANGE_ENABLE=0         : disable target swap and run without exchange.
TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-0}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi

V3_DISABLE_ATTN_TAIL_FLAG=""
if [[ "${V3_DISABLE_ATTN_TAIL}" == "1" ]]; then
  V3_DISABLE_ATTN_TAIL_FLAG="--undetermined_v3_disable_attn_tail_for_motion"
fi

V3_DISABLE_PREV_TARGET_FLAG=""
if [[ "${V3_DISABLE_PREV_TARGET}" == "1" ]]; then
  V3_DISABLE_PREV_TARGET_FLAG="--undetermined_v3_disable_prev_target_for_motion"
fi

RANDOMIZE_INITIAL_POSITIONS_FLAG=""
if [[ "${RANDOMIZE_INITIAL_POSITIONS}" == "1" ]]; then
  RANDOMIZE_INITIAL_POSITIONS_FLAG="--randomize_robot_initial_positions"
fi

V3_REWARD_ENABLE_FLAG=""
if [[ "${V3_REWARD_ENABLE}" == "1" ]]; then
  V3_REWARD_ENABLE_FLAG="--undetermined_v3_reward_enable"
fi

echo "ROOT=${ROOT}"
echo "ABLATION_MODE=no_pretrained_latent"
echo "ROBOT_INITIAL_SPAWN_MODE=${ROBOT_INITIAL_SPAWN_MODE}"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "V3_DISABLE_ATTN_TAIL=${V3_DISABLE_ATTN_TAIL}"
echo "V3_DISABLE_PREV_TARGET=${V3_DISABLE_PREV_TARGET}"
echo "RANDOMIZE_INITIAL_POSITIONS=${RANDOMIZE_INITIAL_POSITIONS}"
echo "V3_REWARD_ENABLE=${V3_REWARD_ENABLE}"
echo "V3_REWARD_M_DROP_SCALE=${V3_REWARD_M_DROP_SCALE}"
echo "V3_REWARD_S_DROP_SCALE=${V3_REWARD_S_DROP_SCALE}"
echo "V3_REWARD_TRAVEL_PENALTY_SCALE=${V3_REWARD_TRAVEL_PENALTY_SCALE}"
echo "V3_SELECTOR_UNIQUE_BONUS_SCALE=${V3_SELECTOR_UNIQUE_BONUS_SCALE}"
echo "V3_SELECTOR_PENDING_PENALTY_SCALE=${V3_SELECTOR_PENDING_PENALTY_SCALE}"
echo "V3_SELECTOR_DUPLICATE_PENALTY_SCALE=${V3_SELECTOR_DUPLICATE_PENALTY_SCALE}"
echo "V3_SELECTOR_PROGRESS_BONUS_SCALE=${V3_SELECTOR_PROGRESS_BONUS_SCALE}"
echo "UNDET_V2_SL_DENSE_SCALE=${UNDET_V2_SL_DENSE_SCALE}"
echo "UNDET_V2_SL_DELTA_SCALE=${UNDET_V2_SL_DELTA_SCALE}"
echo "UNDET_V2_SL_SUCCESS_SCALE=${UNDET_V2_SL_SUCCESS_SCALE}"
echo "UNDET_V2_SL_SUCCESS_THRESHOLD=${UNDET_V2_SL_SUCCESS_THRESHOLD}"
echo "ND_ARRIVAL_REWARD=${ND_ARRIVAL_REWARD}"
echo "ND_GOAL_TERMINAL_REWARD=${ND_GOAL_TERMINAL_REWARD}"
echo "UNDET_V3_TARGET_KL_COEF=${UNDET_V3_TARGET_KL_COEF}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "Python=$(command -v python)"

python train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
  ${V3_DISABLE_ATTN_TAIL_FLAG} \
  ${V3_DISABLE_PREV_TARGET_FLAG} \
  ${RANDOMIZE_INITIAL_POSITIONS_FLAG} \
  ${V3_REWARD_ENABLE_FLAG} \
  --robot_initial_spawn_mode "${ROBOT_INITIAL_SPAWN_MODE}" \
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
  --undetermined_v3_target_kl_coef "${UNDET_V3_TARGET_KL_COEF}" \
  --entropy_coef "${ENTROPY_COEF}" \
  "$@"

date
