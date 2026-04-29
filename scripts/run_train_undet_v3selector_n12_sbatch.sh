#!/bin/bash
# Submit from MAPPO root:
#   cd /home/wangdx_lab/cse12211818/MAPPO && sbatch scripts/run_train_undet_v3selector_n12_sbatch.sh
#
# Pipeline:
#   pretrained selector (from undet_v3_target_latent_decoupled_rank n11) -> MAPPO motion policy (v3 branch)

#SBATCH -o swarm.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J mappo-v3selector-n11copy
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

# Fixed latent checkpoint (do not fallback / auto-switch).
LATENT_MODEL_DIR="../undet_v3_target_latent_decoupled_rank/checkpoints_decoupled_equal/v3decoupled_equal_n11 copy.pt"
LATENT_TRAIN_MODE="${LATENT_TRAIN_MODE:-motion_only}"
ROBOT_INITIAL_SPAWN_MODE="${ROBOT_INITIAL_SPAWN_MODE:-random_box}"
# Target-swap mode switch (v3):
# - TARGET_EXCHANGE_ENABLE=1 (default): enable mutual target swap policy.
# - TARGET_EXCHANGE_ENABLE=0         : disable target swap and run without exchange.
# Usage:
#   TARGET_EXCHANGE_ENABLE=0 sbatch scripts/run_train_undet_v3selector_n12_sbatch.sh
TARGET_EXCHANGE_ENABLE="${TARGET_EXCHANGE_ENABLE:-0}"
TARGET_EXCHANGE_FLAG=""
if [[ "${TARGET_EXCHANGE_ENABLE}" == "1" ]]; then
  TARGET_EXCHANGE_FLAG="--enable_undetermined_v3_exchange"
fi

if [[ ! -f "${LATENT_MODEL_DIR}" ]]; then
  echo "[error] latent checkpoint not found: ${LATENT_MODEL_DIR}" >&2
  echo "[hint] required fixed checkpoint is missing; please generate/restore it first." >&2
  exit 1
fi

echo "ROOT=${ROOT}"
echo "LATENT_MODEL_DIR=${LATENT_MODEL_DIR}"
echo "LATENT_TRAIN_MODE=${LATENT_TRAIN_MODE}"
echo "ROBOT_INITIAL_SPAWN_MODE=${ROBOT_INITIAL_SPAWN_MODE}"
echo "TARGET_EXCHANGE_ENABLE=${TARGET_EXCHANGE_ENABLE}"
echo "Python=$(command -v python)"

python train.py \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v3 \
  --architecture_mode attn_undetermined_goal \
  --use_attn_comm_actor \
  ${TARGET_EXCHANGE_FLAG} \
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
  --undet_v3_target_latent_model_dir "${LATENT_MODEL_DIR}" \
  --undet_v3_latent_train_mode "${LATENT_TRAIN_MODE}" \
  "$@"

date
