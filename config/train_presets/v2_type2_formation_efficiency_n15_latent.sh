#!/usr/bin/env bash
#SBATCH -o swarm.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J swarm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

date
python train.py \
  --train_font_pattern_length 10 \
  --train_font_pattern_policy all \
  --enable_undetermined_goal \
  --enable_undetermined_goal_v2 \
  --undetermined_v2_type2_formation_efficiency \
  --undetermined_v2_goal_slots 10 \
  --undet_v2_head_arch pair_mlp \
  --undetermined_target_embed_dim 32 \
  --undet_v2_pair_mlp_hidden 384 \
  --undet_v2_target_latent_model_dir "../undet_v2_target_latent/checkpoints/selector_n15.pt" \
  --undet_v2_latent_train_mode motion_only \

