Works based on *Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance*.

## Implemented

- Seperated Reward
- Seperated Dataset
- Data Generator
- **Undetermined goal v2**: nearest-goal slot observation pack, `UndeterminedTargetHeadV2`, auction + Hungarian shaping (see below)
- **Undetermined v2_exchange** (optional): pairwise **target_id** swaps inside a local radius when the swap lowers the **fleet** bottleneck \(M=\max_k d(\text{robot}_k,\text{goal}_{\text{target}_k})\) by at least **`--undetermined_v2_exchange_min_gain`**, to shorten the worst agent’s remaining distance to its assigned goal (see below)

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
| `--enable_undetermined_v2_exchange` | Optional; requires v2 and is **incompatible** with v3 (see **v2_exchange** subsection below). |

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

### v2_exchange (optional heuristic target swap)

**v2 only** — cannot be used with `--enable_undetermined_goal_v3`. When enabled, each env `step()` runs **after** `_undetermined_comm_conflict_auction()` and tries **pairwise swaps** of discrete `target_id` between two robots that lie in each other’s **exchange domain** (center-to-center distance ≤ radius). Let \(M_{\text{before}}=\max_k d(p_k, g_{\text{target}_k})\) over agents (collision/success agents contribute 0). For a candidate pair \((i,j)\), let \(M_{\text{after}}\) be the same max after **only** \(i\) and \(j\) exchange targets (all other agents unchanged). A swap is applied only if

**\(M_{\text{before}} - M_{\text{after}} >\) `--undetermined_v2_exchange_min_gain`** (meters). This targets the **worst-off** agent by current distance-to-assigned-goal. Candidate pairs are sorted by that fleet gain (largest first); **disjoint** pairs are taken greedily, at most **`--undetermined_v2_exchange_max_pairs_per_step`** per step. After any swap: `_undetermined_sync_all_goals()`, duplicate auction, and Hungarian shaping are refreshed; both agents’ **`undetermined_target_pending`** are cleared for that pair.

| Parameter | Default | Role |
|-----------|---------|------|
| `--enable_undetermined_v2_exchange` | off | Turn on the heuristic. Requires **`--enable_undetermined_goal_v2`**. |
| `--undetermined_v2_exchange_radius` | `None` → **`--undetermined_comm_radius`** | Domain radius (m): only pairs with distance ≤ this are considered. |
| `--undetermined_v2_exchange_min_gain` | `0.05` | Minimum reduction \(M_{\text{before}}-M_{\text{after}}\) of the fleet max distance-to-assigned-goal (m). |
| `--undetermined_v2_exchange_max_pairs_per_step` | `1` | Cap on disjoint swaps per env step. |
| `--undetermined_v2_exchange_ignore_pending` | off | If set, pairs may swap even when one or both agents have **`undetermined_target_pending`**; default skips any agent that is pending. |

**Code**: `envs/env_core.py` — `_undetermined_v2_exchange_heuristic()`. **CLI checks**: `train.py` (`parser_args`) rejects v3 + exchange together.

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
- Target application / auction: `apply_undetermined_targets`, `_undetermined_auction_duplicate_targets`, optional `_undetermined_v2_exchange_heuristic` (v2 exchange mode).
- Head: `policy/mappo/undetermined_target_head.py` (`UndeterminedTargetHeadV2`).
- Actor wiring: `policy/mappo/actor_critic.py` (`R_Actor`).
- Rollout resolution: `runner/shared/env_runner.py` (`_undetermined_resolve`).
- Dimensions / floors: `config/config.py` (`compute_undetermined_v2_robot_obs_dim`, `apply_undetermined_v2_reward_floors`).
- Pretrained head load: `policy/mappo/undet_v2_latent_ckpt.py`, `runner/shared/base_runner.py` (after `restore`).

### Training (`MAPPO/train.py`)

**预设脚本**（相对 MAPPO 根解析 `../undet_v2_target_latent/...`）：`config/train_presets/v2_type2_formation_efficiency_n15_latent.sh` — undetermined **v2**、`--undetermined_v2_type2_formation_efficiency`、**不**开 `v2_exchange`，`pair_mlp` + **`selector_n15.pt`**（`--undetermined_target_embed_dim 32` 与 `eval_selector_n15.json` 中 `model_layout` 对齐）。**智能体数 `N`** 由 **`--train_font_pattern_length`**（默认 **10**，与 `dataset/10.json` 一致）决定，**不要求** `dataset/15.json`：`n15` 只表示 latent 预训练时的规模，MAPPO 里 head 按 **M 槽 / embed** 加载，可与当前 **N** 不同。需要 **N=20** 时在命令末尾加 `--train_font_pattern_length 20`（并存在 `dataset/20.json`）。在 **`MAPPO/`** 下执行：`./config/train_presets/v2_type2_formation_efficiency_n15_latent.sh`（或 `sbatch`；脚本需 **LF** 换行）。

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

# Same as above plus v2_exchange (heuristic swaps in comm-radius domain)
python train.py --train_font_pattern_length 10 \
  --enable_undetermined_goal --enable_undetermined_goal_v2 --enable_undetermined_v2_exchange

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
| **`--sl-threshold`** | **\(S_L\)** 相似度成功阈值，默认 **`0.97`**。与 `--laplacian-threshold` 独立。 |

**\(S_L\)** 度量（与 `--laplacian-sim-metric` 无关，始终基于同一对 \(\hat{L}, L_{\mathrm{des}}\) 计算）：

\[
S_L = 1 - \frac{\|\hat{L} - L_{\mathrm{des}}\|_F}{\|L_{\mathrm{des}}\|_F}
\]

当 \(\hat{L} = L_{\mathrm{des}}\) 时 \(S_L = 1\)；误差大于 \(\|L_{\mathrm{des}}\|_F\) 时 \(S_L\) 可为负。**相似度成功（\(S_L\)）** 指逐步满足 \(S_L \ge\) **`--sl-threshold`**。对应输出：

| 字段 | 含义 |
|------|------|
| **`sl_type2_mean_step_rate`** | **二类（相似度）成功率**（步比）：各回合内「\(S_L\) 达标」步数 / 回合长度，再对回合取平均。 |
| **`sl_type2_episode_rate`** | 至少出现过一次 \(S_L\) 达标的回合占比。 |
| **`sl_type2_mean_timesteps` / `sl_type2_mean_first_step` / `sl_type2_mean_path_length`** | 在 \(S_L\) 阈值下，每回合达标步数均值、首次达标步号（1 起）均值、智能体轨迹长度均值（与 `lap_type2_*` 路径统计口径一致）。 |

**第一类二类指标（`--laplacian-sim-metric`，与 `--laplacian-threshold`）**： Frobenius 余弦 / `rel_frob` 等；成功率列为：

| 字段 | 含义 |
|------|------|
| **`lap_type2_mean_step_rate`** | 步级成功率：各回合内「相似度 ≥ `--laplacian-threshold`」步数 / 回合长度，再对回合平均。 |
| **`lap_type2_episode_rate`** | 回合级成功率：至少出现过一次相似度达标的回合占比。 |

（与上表 **`sl_type2_*`** 并列：前者为拉普拉斯矩阵相似度指标，后者为 **\(S_L\)** 相对 Frobenius 误差指标。）

```bash
cd /path/to/MAPPO
python scripts/summarize_render_results.py --mode laplacian_type2 \
  --laplacian-threshold 0.97 --laplacian-sim-metric cosine01 --sl-threshold 0.97 --csv data.csv
```

### Sibling folder `undet_v2_target_latent`（与 MAPPO 同级）

路径形如 **`../undet_v2_target_latent/`**（与 **`MAPPO/`** 同一父目录）。脚本**不**在 MAPPO 环境里跑 PPO，但与 undetermined v2 **pair_mlp 选择头**预训练衔接：**`train.py`** 默认从 **`MAPPO/config/config.py`** 同步宽度与几何超参；**`test.py`** 在合成协议下评估 **`checkpoints/`** 权重，并支持直接加载 MAPPO **`actor.pt`** 中的 **`undetermined_head.*`**。

#### 1. 槽位预训练：`undet_v2_target_latent/train.py`

在 **合成 2D 指派** 上训练 **`TargetLatentSelector`**（与 MAPPO **`--undet_v2_head_arch pair_mlp`** 下 **`UndeterminedTargetHeadV2PairMLP`** 同构；权重可经 **`--undet_v2_target_latent_model_dir`** 加载）。观测与 undetermined v2 packing 对齐（见该目录 **`pack_obs.py`**）。

**匈牙利监督 `M0`（默认面向 bottleneck）**：默认 **`--hungarian-objective bottleneck`**，在代价矩阵上用 **欧氏距离** 做 **minimax** 指派（最小化「最远一对」边长），与 MAPPO 侧 **`v2_exchange` / 编队公平性** 更一致；旧行为 **`--hungarian-objective sum`**（最小总平方距离）。`run.sh` 可用环境变量 **`HUNGARIAN_OBJECTIVE=sum`** 覆盖。checkpoint 的 **`model_layout.hungarian_objective`** 会记录所用目标。

**与 MAPPO 默认对齐（推荐）**：`train.py` 默认从 **MAPPO 仓库** 的 **`config/config.py`** 解析超参并写入本脚本的 **`--goal-slots` / `--hidden` / `--d-emb` / `--obs-goal-radius` / `--box` / LayerNorm** 等（实现见 **`mappo_defaults.py`**）。MAPPO 与 `undet_v2_target_latent` **同级**时无需额外配置；否则传入 **`--mappo-root /path/to/MAPPO`**。若要用脚本内手写默认、不从 MAPPO 拉取：**`--no-from-mappo`**。

**交互式启动**（必须先 `cd` 到 **`undet_v2_target_latent`**，保证相对路径与 Slurm 一致）：

```bash
cd /path/to/cse12211818/undet_v2_target_latent

# 使用 MAPPO 默认（含 pair_mlp 相关宽度）；仅改合成规模与步数示例：
python train.py --mappo-root ../MAPPO --n 10 --steps 8000

# 完全手写超参（关闭从 MAPPO 同步）：
python train.py --no-from-mappo --n 10 --goal-slots 10 --hidden 384 --d-emb 96 --steps 8000
```

**Slurm 批作业（`run.sh`）**：脚本依赖环境变量 **`SLURM_SUBMIT_DIR`** 定位工程目录，因此 **必须在 `undet_v2_target_latent` 目录下提交**，例如：

```bash
cd /path/to/cse12211818/undet_v2_target_latent
sbatch run.sh
```

勿在其他工作目录执行 **`sbatch /绝对路径/run.sh`**（除非为该作业设置 **`SLURM_SUBMIT_DIR`** 或使用 **`#SBATCH --chdir=`** 指向本目录）；否则 Slurm 拷贝后的脚本目录下没有 **`train.py`**，会出现 `can't open file '.../train.py'`。

**`train.py` 单次解析**：`--mappo-root` 后必须跟路径；若路径缺失，下一项 **`--n`** 会被 argparse 误吞为 `--mappo-root` 的值，进而报 **`unrecognized arguments: 8`**。`run.sh` 会将 **`MAPPO_ROOT`** 规范为已存在的目录；节点无 CUDA 时可设 **`USE_AMP=0`** 关闭 **`--amp`**。批量评测用同级 **`rerun.sh`**（同样在工程目录下 **`sbatch rerun.sh`**）。

| 参数 | 含义 |
|------|------|
| **`--mappo-root`** | MAPPO 仓库根目录（默认尝试同级 **`../MAPPO`**）。 |
| **`--no-from-mappo`** | 不从 MAPPO 拉默认，改用本脚本 argparse 的静态默认。 |
| `--n` | 合成任务中智能体数 = 目标数（**n×n** 匹配）。 |
| `--goal-slots` | 槽数 **M**；未传且未 `--no-from-mappo` 时与 MAPPO **`--undetermined_v2_goal_slots`** 一致。 |
| `--hungarian-objective` | **`bottleneck`**（默认）或 **`sum`**：监督标签 `M0` 分别为 minimax 欧氏指派 / 总平方代价指派。 |
| `--steps` | 本脚本优化步数（**不是** MAPPO 的 `num_env_steps`）。 |
| **`--save-dir`** | 权重目录（默认 **`checkpoints/`**）。 |
| **`--save-name`** | 权重文件名（默认 **`target_latent_selector.pt`**）。MAPPO 默认会在 **`../undet_v2_target_latent/checkpoints/`** 下查找该名或 **`selector_n*.pt`** 的拷贝策略见 **`run.sh`** 末尾注释。 |

**与 MAPPO「模式」的关系**：合成 **`train.py`** 不跑 MAPPO 环境，**不依赖** `architecture_mode` / `use_attn_comm_actor` 等；但 **宽度类超参** 应与 MAPPO **`pair_mlp`** 及 **`undetermined_v2_goal_slots`** 一致，便于加载到 **`undetermined_head.*`**。点积头 **`UndeterminedTargetHeadV2`** 与 **`TargetLatentSelector`** 结构不同，需 MAPPO 侧 **`--undet_v2_head_arch pair_mlp`** 才能直接加载 **`target_latent_selector.pt`**。

其余 CLI 见该目录 **`train.py`** argparse。训练后可用 **`test.py`** 评估同一 **`checkpoints/`** 路径；**`test.py`** 也可直接加载 MAPPO **`actor.pt`**（仅含 **`undetermined_head.*`** 时），见下一节。

#### 2. 选择头评估：`test.py`

**定位**：评测 **`TargetLatentSelector`**；默认读 **同目录 `checkpoints/`** 下由 **`train.py`** 写出的 **`model`** 块。合成协议与 **`train.py`** 相同：**`[0, box]²`**、匈牙利 **`M0`**、**`batch_pack`**、CE / greedy / Hungarian 一致率等。也可传入 MAPPO **`actor.pt`**：若 state dict 含 **`undetermined_head.*`**（**`pair_mlp`** 头），脚本会剥前缀并推断 **`model_layout`**；此时 **`obs_goal_radius` / `box`** 可从 MAPPO 配置读取（**`--mappo-root`**）。

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
