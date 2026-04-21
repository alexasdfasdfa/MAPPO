Works based on *Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance*.

## Implemented

- Seperated Reward
- Seperated Dataset
- Data Generator
- **Undetermined goal v2**: nearest-goal slot observation pack, `UndeterminedTargetHeadV2`, auction + Hungarian shaping (see below)

## TODO

- Faster Code

---

## Undetermined goal v2 (`enable_undetermined_goal_v2`)

Undetermined mode means each robot keeps a **sticky** discrete goal index; when goals are (re)assigned, the policy’s **target head** scores candidates and the environment runs **auction / pending** logic. **v2** changes the observation and reward profile relative to v1: the robot row uses a **fixed width** of **M nearest goals** in world space instead of encoding all **K** goals and **K−1** peer target ids (v1 scales with swarm size **K**).

### Enabling and exclusivity

| Requirement | Notes |
|-------------|--------|
| `--enable_undetermined_goal` | Master switch for undetermined mode (mutually exclusive with `--enable_dynamic_goal_assignment`). |
| `--enable_undetermined_goal_v2` | v2 obs + v2 reward floors (`apply_undetermined_v2_reward_floors` in `train.py`). Implies v1 undetermined is on. |

Training also sets `robot_obs_dim` from **`compute_undetermined_v2_robot_obs_dim(M)`** where **M = `--undetermined_v2_goal_slots`** (runtime packing still caps **M ≤ K** goals in the env). **`config/config.py` defaults**: **`--undetermined_v2_goal_slots`** = **10**; **`--num_attention_agents`** = **10** (obs rows for hybrid AttnComm / neighbor packing; **`attn_comm_ally_slots`** still defaults from **`neighbor_n`**, also **10**). **`train.py`** `main()` already forces **`num_attention_agents = 10`**.

### Robot observation row layout (v2 core)

Per-robot vector length **`robot_obs_dim + 2`**: env builds **`robot_obs_dim`** scalars then appends **`px, py`**.

Layout (indices are 0-based in the packed row **before** px, py; the actor uses the full length including px, py as the last two channels):

| Block | Length | Content |
|--------|--------|---------|
| Base | 7 | `(gx - px, gy - py, v, θ, for_feature, vx_formation, vy_formation)` — offset to **assigned** goal and kinematics / formation hints. |
| Nearest slots | **5 × M** | For each slot **m ∈ [0, M−1]** (nearest goals by distance): `(dx/scale, dy/scale, in_r, cobs, k_norm)` with `scale = max(undetermined_obs_goal_radius, 1)`; padding uses `k_norm = -1` (masked in the head). |
| Pending | 1 | `1` if this agent still needs a target resolution pass, else `0`. |
| Position | 2 | `px, py` (world). |

**M** is **`--undetermined_v2_goal_slots`** in config; at most **K** slots carry real goals (**K** = number of goal targets / pattern size).

### Policy structure

- **`UndeterminedTargetHeadV2`** (`policy/mappo/undetermined_target_head.py`): `ego_mlp` on **8** dims (base 7 + pending), `goal_mlp` on **5** dims per slot, **dot-product logits** over **M** slots; invalid slots get large negative logits.
- **Embedding width**: `--undetermined_target_embed_dim` (default 32).
- **Without** `--use_attn_comm_actor`: the motion trunk is **`MLPBase` on the full robot row**; target logits are **not** part of the PPO action log-prob graph during rollout (`get_undetermined_targets` runs under `torch.no_grad`). For **finetune** of the head when loading a pretrained head, training adds an **auxiliary supervised slot loss** (see `--undet_v2_target_head_aux_coef`).
- **With** `--use_attn_comm_actor` and **`--architecture_mode attn_undetermined_goal`**: **hybrid** row = **same v2 core** + **AttnComm tail** (allies, received messages, humans, obstacle block). Width: `compute_undetermined_v2_attn_hybrid_robot_obs_dim(M, P, H, message_dim)` in `config/config.py`. This path requires attn comm on the CLI (`train.py` checks).

### Target head warm-start (non-AttnComm only)

| Flag | Role |
|------|------|
| `--undet_v2_target_latent_model_dir` | Directory containing `models/actor.pt` (or a direct `*.pt`). After optional `model_dir` restore, loads **only** `undetermined_head.*` weights. |
| `--undet_v2_latent_train_mode` | `motion_only`: freeze head (no optimizer step). `finetune_all`: train head + motion (default). |
| `--undet_v2_target_head_aux_coef` | Weight on auxiliary cross-entropy aligning the head to the slot that matches the goal implied by obs; used in **`finetune_all`** when coef > 0. Ignored in **`motion_only`**. |

`motion_only` requires **`--undet_v2_target_latent_model_dir`** to be set.

### Auction and shaping (shared with v1 unless noted)

| Parameter | Default (config) | Role |
|-----------|------------------|------|
| `--undetermined_obs_goal_radius` | 5.0 | Within this radius, per-goal **claimed** observation is resolved for slots. |
| `--undetermined_comm_radius` | 6.0 | Local radius for **same-target** auction tie-break. |
| `--undetermined_max_auction_rounds` | 8 | Max rounds of **pending** resolution per env step. |
| `--undetermined_hungarian_reward_scale` | 0.10 | Scale for Hungarian assignment-gap shaping (capped further under v2 floors). |
| `--undetermined_v2_hungarian_team_divisor` | 8.0 | v2: divide Hungarian bonus by this constant instead of by **N** agents. |

### v2-specific reward / discount knobs

These tune **`apply_undetermined_v2_reward_floors`** (invoked from `train.py` when v2 is on), which builds on v1 floors then adjusts avoid / nav / goal discounts, distance penalties, terminals, etc.

| Parameter | Default | Role (short) |
|-----------|---------|----------------|
| `--undetermined_v2_discount_avoid_mult` | 0.42 | Scales `nd_discount_avoid` after floors. |
| `--undetermined_v2_dist_penalty_mult` | 0.38 | Scales linear + quadratic **undetermined** distance penalties on navigation reward. |
| `--undetermined_v2_avoid_exp_cap` | 2.0 | Caps raw avoid magnitude from exp discomfort (reward calculator). |
| `--undetermined_v2_approach_reward_scale` | 0.85 | Dense approach bonus scale toward assigned goal. |
| `--undetermined_v2_approach_reward_cap` | 0.55 | Cap on per-step approach shaping (world units). |

### Shared undetermined (v1 / v2) navigation knobs

Still read in undetermined runs; v2 floors may clamp them.

| Parameter | Default | Role (short) |
|-----------|---------|----------------|
| `--undetermined_far_goal_progress_dist_thresh` | 4.0 | Beyond this distance to goal, boost progress coefficient. |
| `--undetermined_far_goal_progress_boost` | 1.12 | Multiplier on progress when far (capped under floors). |
| `--undetermined_goal_distance_penalty_scale` | 0.012 | Linear `dist` penalty on `r_nav` (0 = off). |
| `--undetermined_goal_dist_penalty_quad_scale` | 0.00012 | Quadratic `dist²` penalty on `r_nav`. |

### Architecture preset (optional hybrid)

| Parameter | Choices | Role |
|-----------|---------|------|
| `--architecture_mode` | `default`, `attn_undetermined_goal` | Hybrid **ConsMAC-style** tail on top of v2 core obs; requires **`--use_attn_comm_actor`**. |

AttnComm slot counts (`--attn_comm_ally_slots`, `--attn_comm_human_slots`, `--attn_comm_message_dim`, …) are resolved in `resolve_attn_comm_args` when attn is enabled.

### Code map

- Observation packing: `envs/env_core.py` (`_pack_undetermined_v2_obs`, hybrid variant).
- Target application / auction: `apply_undetermined_targets`, `_undetermined_auction_duplicate_targets`.
- Head: `policy/mappo/undetermined_target_head.py` (`UndeterminedTargetHeadV2`).
- Actor wiring: `policy/mappo/actor_critic.py` (`R_Actor`).
- Rollout resolution: `runner/shared/env_runner.py` (`_undetermined_resolve`).
- Dimensions / floors: `config/config.py` (`compute_undetermined_v2_robot_obs_dim`, `apply_undetermined_v2_reward_floors`).
- Pretrained head load: `policy/mappo/undet_v2_latent_ckpt.py`, `runner/shared/base_runner.py` (after `restore`).

### Training (`MAPPO/train.py`)

在 **`MAPPO`** 仓库根目录执行（保证相对路径 `./dataset` 等正确）：

```bash
cd /path/to/MAPPO
python train.py [选项...]
```

**说明**：这里的 **`train.py`** 是 **本仓库 MAPPO** 的入口，与下一节同级目录 **`undet_v2_target_latent/train.py`** 不是同一个文件。

**入口**：`train.py` 使用 `config.get_config()` 解析全部超参，再经 **`parser_args`** 做模式相关校验与 **`robot_obs_dim`** 等推导；额外业务参数在 `parser_args` / `main()` 里写死一部分（与命令行叠加时注意下文说明）。

**智能体数与字形数据**：`parser_args` 会把 **`num_agents` 强制对齐为 `--train_font_pattern_length`**（与 `dataset/<N>.json` 的 **N** 一致）。训练前需存在 **`MAPPO/dataset/<N>.json`**。字形池由 `--train_font_pattern_policy` / `--train_font_pattern_names` 等控制（见 `config/config.py` 与 `envs/env_core.py` 的 `FontPatternLoader`）。

**`main()` 中覆盖的默认（会覆盖你在命令行传入的同名项）**：`num_humans=2`，`num_attention_agents=10`，`n_rollout_threads=40`，`episode_length=400`，`num_env_steps = n_rollout_threads * episode_length * 1200`，`num_mini_batch=400`，`save_interval=1`，`log_interval=1`。需要改规模时可直接编辑 **`train.py`** 里对应赋值。

**`model_dir`（续训）**：`main()` 里将 **`all_args.model_dir = None`**，即从随机初始化开训；若要从已有 **`actor.pt` / `critic.pt`** 继续，需在 **`train.py`** 的 `main()` 中把该行改为指向某次 run 的 **`.../results/train/run*/models`** 目录（或取消注释脚本中的示例路径）。命令行传入的 `--model_dir` 会在进入 `main()` 后被上述赋值覆盖。

**输出目录**：自动在 **`MAPPO/results/train/`** 下新建递增的 **`run1`, `run2`, …**；每次运行写入 **`run_flags.txt`**（记录关键开关与 `num_agents` 等）、**`models/actor.pt`**、**`models/critic.pt`**、**`logs/`**（TensorBoard）。

**算法**：默认配置里 **`algorithm_name`** 等需满足 `train.py` 中的断言（例如 **rmappo** 要求开启 recurrent policy）。其余 PPO / MAPPO 相关参数见 **`config/config.py`**。

**示例（undetermined v2 + hybrid attn，仅作参数组合示意）**：

```bash
python train.py --train_font_pattern_length 10 \
  --enable_undetermined_goal --enable_undetermined_goal_v2 \
  --architecture_mode attn_undetermined_goal --use_attn_comm_actor

python train.py --n 20 --goal-slots 20 --steps 8000 \
  --save-name 20.pt\
  --save-dir ./checkpoints 
```

### Render 回放统计（`scripts/summarize_render_results.py`）

对 **`results/render/run*/`** 下由 **`EnvRunner.render()`** 写出的 **`succ/`**、**`coords/`** 与 **`episode_meta.json(l)`** 做汇总。默认模式统计抵达目标等轨迹指标；**`--mode laplacian_type2`** 会额外统计基于**图拉普拉斯**的 **相似度成功**（与「抵达 assigned target」的 **目标成功**区分）。

| 称呼 | 含义 |
|------|------|
| **目标成功**（一类） | 各智能体是否到达其目标：来自 **`succ/success_agent*.txt`** 的逐步标志（与环境中 `reach_goal` 一致）。 |
| **相似度成功** | 当前队形拉普拉斯 \( \hat{L} \) 与由目标位置构造的期望拉普拉斯 \( L_{\mathrm{des}} \) 的**相似度**不低于阈值；表示队形与期望图结构对齐，**不**等同于已抵达目标点。 |

**离线重建方式（与 `envs/env_core.py` 中 `step` 一致）**：边权为智能体间位置差的平方和 **`get_weight`**，得到 \(W\) → \(L=D-W\) → 对称归一化得到 \( \hat{L} \)；\( L_{\mathrm{des}} \) 由 **`episode_meta`** 中 **`agent_goals`**（优先）或长度等于 **`n`** 的 **`goal_positions`** 在**回合起始**构造（若目标在回合内移动，此处不逐步更新）。

**CLI 要点**

| 参数 | 说明 |
|------|------|
| **`--mode laplacian_type2`** | 启用相似度成功相关列。 |
| **`--laplacian-threshold`** | 相似度阈值，默认 **`0.97`**（约「大于 97%」的刻度，视下方指标而定）。 |
| **`--laplacian-sim-metric`** | **`cosine01`**（默认）：Frobenius 余弦映射到 \([0,1]\)，即 \((\cos+1)/2\)，便于与 **0.97** 对照；**`cosine`**：余弦本身 \([-1,1]\)；**`rel_frob`**：\(1 - \| \hat{L}-L_{\mathrm{des}}\|_F^2 / (\|\hat{L}\|_F^2+\|L_{\mathrm{des}}\|_F^2)\)。 |

**`laplacian_type2` 输出字段（节选）**：`lap_type2_mean_timesteps`（每回合中相似度成功的步数均值）、`lap_type2_mean_first_step`（首次达到相似度成功的步号均值）、`lap_type2_mean_path_length`（有效回合上智能体轨迹长度均值）、`lap_type2_episodes_used` / `lap_type2_episodes_skipped`。

```bash
cd /path/to/MAPPO
python scripts/summarize_render_results.py --mode laplacian_type2 \
  --laplacian-threshold 0.97 --laplacian-sim-metric cosine01
```

### Sibling folder `undet_v2_target_latent`（与 MAPPO 同级）

路径形如 **`../undet_v2_target_latent/`**（与 **`MAPPO/`** 同一父目录）。该目录下的脚本**不**属于 MAPPO 包内训练循环，但与 undetermined v2 **槽位 / 选择头**工作流相关：**`train.py`** 监督训练 **`TargetLatentSelector`**，**`test.py`** 在同一合成协议下评估 **`checkpoints/`** 里保存的权重（**不**加载 MAPPO、**不**需要 MAPPO 命令行参数）。

#### 1. 槽位预训练：`train.py`

在 **合成 2D 指派** 上训练槽位选择器（观测与 undetermined v2 槽位 packing 对齐；细节见该目录代码与 `pack_obs.py`）。**启动示例**（在该目录下执行）：

```bash
cd ../undet_v2_target_latent
python train.py --n 20 --goal-slots 20 --steps 8000
```

| 参数 | 含义 |
|------|------|
| `--n` | 智能体数 = 目标数（合成 **n** 对 **n** 匹配）。 |
| `--goal-slots` | 槽数 **M**，应对齐 MAPPO 的 **`--undetermined_v2_goal_slots`**（latent `train.py` 默认 **10**）。 |
| `--steps` | 该脚本内的优化步数（**不是** MAPPO 环境里的 `num_env_steps`）。 |
| **`--save-dir`** | 权重所在目录（默认 **`checkpoints/`**）。 |
| **`--save-name`** | 权重**文件名**（仅 basename，写在 `--save-dir` 下；默认 **`target_latent_selector.pt`**）。 |

**与 MAPPO「模式」的关系**：`train.py` 只在合成指派上训练 **`TargetLatentSelector`**，**不读取、不依赖** MAPPO 的 `architecture_mode`、`use_attn_comm_actor`、`enable_undetermined_goal` 等运行模式；脚本内打印的匹配 / 准确率等指标即选择头层面的自测，**同样**无需切换上述 MAPPO 开关。与 MAPPO 的衔接仅要求 **维数类超参一致**（如 **M**、嵌入维、隐藏层宽度），以便后续把权重写入 MAPPO 的 **`UndeterminedTargetHeadV2`**（例如 **`--undet_v2_target_latent_model_dir`**）。

其余 CLI 见该目录 **`train.py`** 的 argparse。训练结束后可用 **`test.py`**（下一节）对 **`--save-dir`/`--save-name`** 对应路径（默认 **`checkpoints/target_latent_selector.pt`**）做离线批量评估。

#### 2. 选择头评估：`test.py`

**定位**：只测 **同目录 `checkpoints/`** 里由 **`train.py`** 写出的权重（`TargetLatentSelector`）。合成协议与 **`train.py`** 相同：**`[0, box]²`** 采样、匈牙利 **`M0`**、**`batch_pack`**、前向 logits 与 CE / greedy / Hungarian 一致率等指标。**不**导入 MAPPO。

**模型宽度与 `#agents` 解耦**：新 checkpoint 含 **`model_layout`**——固定 **`pack_goal_slots`（M）**、**`packed_obs_dim`**（与 `pack_obs.obs_dim(M)` 一致）、**`selector_hidden` / `selector_d_emb` / `selector_use_layernorm`**。**`test.py` 仅用 `model_layout` 构建网络**；评测时的智能体数 **`eval_num_agents`** 由 **`-n`** 指定，或与训练解耦地回退到 **`args.n`**（旧 ckpt）。**`batch_pack(..., goal_slots=)` 始终用 `pack_goal_slots`**，不再用「当前 n」去推断槽位数。

**默认检查点**：`<undet_v2_target_latent>/checkpoints/target_latent_selector.pt`。无 **`model_layout`** 的旧文件：从 **`args`** 推断布局（与早期行为兼容）。

**程序结构**：CLI → **`torch.load`** → **`get_model_layout`** → **`TargetLatentSelector`** + **`load_state_dict`** → **`args`** 仅提供采样/损失等（`box`、`tau_ce`、`batch_size`…）→ **`run_eval(..., n=eval_num_agents, pack_goal_slots=layout)`**。

**命令行**（布局不进 CLI）：

| 参数 | 说明 |
|------|------|
| **`--checkpoint`** | 权重 **`.pt`**；默认 **`checkpoints/target_latent_selector.pt`**。 |
| **`--eval-batches`** | 随机 batch 数（默认 64）。 |
| **`--out`** | 指标 JSON。 |
| **`-n` / `--n`** | 一个或多个整数 **`N`**（`nargs` 可变长）：按顺序对 **每个** `N` 各跑一轮 **`eval_batches`**；**与模型槽位 M 独立**。省略且 **`args.n`** 存在时，等价于 **`-n <args.n>`** 一轮；否则须传 **`-n`**。 |

**示例**：

```bash
cd ../undet_v2_target_latent

python test.py \
  --checkpoint ./checkpoints/target_latent_selector.pt \
  --eval-batches 128 \
  --out ./head_metrics.json

# 与训练不同 swarm 规模下 smoke 同一套槽位权重（需与 M、box 等匹配）：
python test.py --checkpoint ./checkpoints/target_latent_selector.pt -n 32 --eval-batches 64

# 一次评测多个 n（输出 JSON 含 results 数组；仅一个 n 时另有 latest 字段）：
python test.py --checkpoint ./checkpoints/target_latent_selector.pt -n 8 16 32 --out multi_n.json
```
