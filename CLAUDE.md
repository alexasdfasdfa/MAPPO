# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent PPO (MAPPO) for robot swarm formation control with collision avoidance. Based on *Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance*. The project trains a centralized policy for N robots to navigate to goal positions (letter "font patterns") while avoiding collisions with each other and with human-controlled agents.

## Key Commands

### Environment Setup

```bash
# Create conda environment
bash scripts/create_swE2_env.sh

# Install dependencies (includes RVO2 C library)
pip install -r requirements.txt
# Or install RVO2 manually:
bash scripts/build_rvo2.sh
bash scripts/install_python_rvo2.sh
```

### Training

```bash
cd /path/to/MAPPO

# Quick start: undetermined v2 + target exchange (preset)
bash scripts/run_train_undet_v2_exchange.sh

# Undetermined v2 only
bash scripts/run_train_undet_v2.sh

# Manual launch
python train.py --train_font_pattern_length 10 \
  --enable_undetermined_goal --enable_undetermined_goal_v2 \
  --enable_undetermined_v2_exchange \
  --undet_v2_head_arch pair_mlp
```

Training outputs go to `results/train/runN/` with models, TensorBoard logs, and `run_flags.txt`.

### Rendering / Visualization

```bash
# Render trained model
bash scripts/run_render.sh -- --model_dir results/train/runN/models

# Render with v2 exchange
bash scripts/run_render_undet_v2_exchange.sh -- --model_dir results/train/runN/models

# Summarize render results
python scripts/summarize_render_results.py --mode laplacian_type2 --csv data.csv
```

### Latent Selector Pre-training (sibling folder)

The `../undet_v2_target_latent/` directory contains a separate pre-training pipeline for the target selection head:

```bash
cd ../undet_v2_target_latent
python train.py --mappo-root ../MAPPO --n 10 --steps 8000
python test.py --checkpoint ./checkpoints/target_latent_selector.pt -n 10
```

## Architecture

### High-Level Structure

```
train.py                  # Main training entry point
render.py                 # Model visualization / replay
config/config.py          # All hyperparameters, observation dimension calculators, reward floor functions
envs/
  env_core.py             # Core simulation: robots, humans, RVO2 collision, font pattern goals
  env_discrete.py         # Gym wrapper: DiscreteActionEnv (dir + vel actions, optional goal selection)
  env_continuous.py       # Continuous action variant
  env_wrappers.py         # DummyVecEnv / SubprocVecEnv for parallel rollouts
  utils/
    reward_calculator.py  # Centralized reward computation
    exchange_network.py   # Optional exchange network for data collection
    hungarian_opt.py      # Optimal assignment (Hungarian algorithm)
    robot.py / human.py   # Agent classes
policy/
  mappo/
    actor_critic.py       # R_Actor / R_Critic networks
    MAPPOPolicy.py        # Policy wrapper
    rmappo.py             # RMAPPO training algorithm
    undetermined_target_head.py  # Target selection heads (V1, V2 dot_product/pair_mlp, V3 attention)
    utils/                # MLP, RNN, LSTM, attention, distributions, POPART
runner/
  shared/
    base_runner.py        # Runner: buffer setup, policy update, checkpoint save/restore
    env_runner.py         # Training loop: rollout, target resolution, logging
  separated/              # Non-shared variant (not commonly used)
dataset/
  10.json / 20.json       # Font pattern definitions (letter-shaped goal formations)
scripts/                  # Shell scripts for training, rendering, environment setup
```

### Core Data Flow

1. **train.py** parses args via `config.get_config()`, creates parallel envs (`SubprocVecEnv`), initializes actor/critic networks, and runs the shared `Runner`.
2. **env_core.py** (`EnvCore`) is the simulation engine: maintains robots/humans, computes RVO2 collision-free velocities, evaluates Laplacian formation similarity, and packs observations.
3. **env_discrete.py** (`DiscreteActionEnv`) wraps EnvCore as a Gym environment, handling the multi-discrete action space (direction + velocity, optionally goal target selection).
4. **actor_critic.py** defines `R_Actor` (policy) and `R_Critic` (value function). The actor can be a plain MLP, or augmented with AttnComm communication or undetermined goal selection heads.
5. **env_runner.py** orchestrates the training loop: collect rollouts, resolve target assignments (auction/Hungarian/exchange), compute rewards, update policy via RMAPPO.

### Observation Modes

| Mode | Flag | Obs Dimension Source |
|------|------|---------------------|
| Static | default | Base 7 dims + px, py |
| Dynamic goals | `--enable_dynamic_goal_assignment` | `compute_dynamic_robot_obs_dim` |
| Undetermined v1 | `--enable_undetermined_goal` | `compute_undetermined_robot_obs_dim` |
| Undetermined v2 | `--enable_undetermined_goal_v2` | `compute_undetermined_v2_robot_obs_dim(M)` |
| Undetermined v2 + AttnComm hybrid | `--architecture_mode attn_undetermined_goal` | `compute_undetermined_v2_attn_hybrid_robot_obs_dim` |
| Undetermined v3 | `--enable_undetermined_goal_v3` | `compute_undetermined_v3_robot_obs_dim` |

### Key Concepts

- **Font Patterns**: Goal formations defined as JSON files in `dataset/`. Each pattern is a letter shape with N coordinate points. The number of agents (`num_agents`) is forced to match `--train_font_pattern_length` and requires a corresponding `dataset/<N>.json`.
- **Undetermined Goals (v2)**: Robots maintain sticky target assignments. Per step: observe M nearest goals, score via `UndeterminedTargetHeadV2`, run auction for conflict resolution, optionally apply v2_exchange pairwise swaps.
- **v2_exchange**: Heuristic target swapping within comm radius. Accept criteria: `fleet_m` (minimize fleet bottleneck), `pair_max` (local pair improvement), `cone_mutual_greedy_m` (velocity cone + greedy).
- **Reward**: Centralized in `reward_calculator.py`. Components: navigation, collision avoidance, formation, goal progress, Hungarian assignment bonus. v2 reward floors adjust scaling.
- **Laplacian Formation**: Graph Laplacian similarity measures how well the swarm matches the target formation shape. Computed via `get_weight` (inter-agent distances) -> W -> L = D - W -> normalized.

## Important Notes

- `train.py` `main()` overrides several defaults (num_humans=2, n_rollout_threads=40, etc.) — direct edits to `main()` may be needed to change these.
- `model_dir` is hardcoded to `None` in `main()`; modify it or pass via CLI for checkpoint resumption.
- The conda environment name is `swE2`.
- RVO2 is a compiled C extension (`rvo2.cpython-*.so`). May need rebuilding on different platforms.
- Scripts use `nohup` via `lib_train_runner.sh` — training runs in background and survives terminal close.
- `results/train/` auto-increments run numbers. Each run gets its own `runN/` directory.
