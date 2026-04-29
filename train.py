# !/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Repo root (not cwd) so Slurm / arbitrary launch dirs still find `runner`, `envs`, etc.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# import rvo2
from config.config import (
    get_config,
    apply_architecture_mode_preset,
    resolve_attn_comm_args,
    resolve_attn_comm_ppo_batch_args,
    resolve_dynamic_target_reasoning_args,
    compute_attn_comm_robot_obs_dim,
    compute_attn_comm_tail_dim,
    compute_dynamic_robot_obs_dim,
    compute_undetermined_robot_obs_dim,
    compute_undetermined_v2_robot_obs_dim,
    compute_undetermined_v2_attn_hybrid_robot_obs_dim,
    compute_undetermined_v3_robot_obs_dim,
    apply_undetermined_reward_floors,
    apply_undetermined_v2_reward_floors,
)
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():

            from envs.env_discrete import DiscreteActionEnv
            # pass env rank into env_core (used for font pattern assignment)
            all_args.env_rank = rank
            env = DiscreteActionEnv(all_args)

            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])    #单线程
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])   #多线程


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():

            from envs.env_discrete import DiscreteActionEnv
            all_args.env_rank = rank
            env = DiscreteActionEnv(all_args)

            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])    #单线程
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])   #多线程


def parser_args(args, parser):
    parser.add_argument(
        "--num_agents",
        type=int,
        default=15,
        help="number of players (training overrides this to match --train_font_pattern_length)",
    )
    parser.add_argument("--random_act_prob", type=int, default=0, help="the probability of robot to choice random action")
    parser.add_argument(
        "--undet_v3_target_latent_model_dir",
        type=str,
        default="",
        help="Optional pretrained selector checkpoint for undetermined v3 head loading.",
    )
    parser.add_argument(
        "--undet_v3_latent_train_mode",
        type=str,
        default="finetune_all",
        choices=["motion_only", "finetune_all"],
        help="When using undet_v3_target_latent_model_dir: freeze selector head (motion_only) or finetune all.",
    )

    all_args = parser.parse_known_args(args)[0]
    apply_architecture_mode_preset(all_args)
    resolve_dynamic_target_reasoning_args(all_args)

    if str(getattr(all_args, "architecture_mode", "default")) == "attn_undetermined_goal" and not getattr(
        all_args, "use_attn_comm_actor", False
    ):
        raise ValueError(
            "--architecture_mode attn_undetermined_goal requires --use_attn_comm_actor (ConsMAC-style comm encoder)."
        )

    # Keep explicit num_agents unless a latent checkpoint requires fixed agent count.
    all_args.num_agents = int(getattr(all_args, "num_agents", 10))

    if getattr(all_args, "enable_undetermined_goal_v2", False) and not getattr(
        all_args, "enable_undetermined_goal", False
    ):
        raise ValueError("--enable_undetermined_goal_v2 requires --enable_undetermined_goal")
    if getattr(all_args, "enable_undetermined_goal_v3", False) and not getattr(
        all_args, "enable_undetermined_goal", False
    ):
        raise ValueError("--enable_undetermined_goal_v3 requires --enable_undetermined_goal")
    if getattr(all_args, "enable_undetermined_goal_v3", False) and getattr(
        all_args, "enable_undetermined_goal_v2", False
    ):
        raise ValueError("--enable_undetermined_goal_v3 cannot be used together with --enable_undetermined_goal_v2")
    if getattr(all_args, "enable_undetermined_v2_exchange", False):
        if not getattr(all_args, "enable_undetermined_goal_v2", False):
            raise ValueError("--enable_undetermined_v2_exchange requires --enable_undetermined_goal_v2")
        if getattr(all_args, "enable_undetermined_goal_v3", False):
            raise ValueError("--enable_undetermined_v2_exchange is incompatible with --enable_undetermined_goal_v3")

    _lat_dir = getattr(all_args, "undet_v2_target_latent_model_dir", None)
    _lat_v3_dir = getattr(all_args, "undet_v3_target_latent_model_dir", None)
    _pair_arch = str(getattr(all_args, "undet_v2_head_arch", "dot_product")) == "pair_mlp"
    if _pair_arch and getattr(all_args, "use_attn_comm_actor", False):
        raise ValueError("--undet_v2_head_arch pair_mlp is incompatible with --use_attn_comm_actor (pure v2 obs only).")
    if _lat_dir and str(_lat_dir).strip():
        if not getattr(all_args, "enable_undetermined_goal", False):
            raise ValueError("--undet_v2_target_latent_model_dir requires --enable_undetermined_goal")
        if getattr(all_args, "enable_undetermined_goal_v3", False):
            raise ValueError(
                "--undet_v2_target_latent_model_dir is incompatible with --enable_undetermined_goal_v3 "
                "(v3 always uses AttnComm hybrid obs)."
            )
        if not getattr(all_args, "enable_undetermined_goal_v2", False):
            raise ValueError("--undet_v2_target_latent_model_dir requires --enable_undetermined_goal_v2")
        if getattr(all_args, "use_attn_comm_actor", False):
            raise ValueError(
                "--undet_v2_target_latent_model_dir is only for runs with --use_attn_comm_actor disabled "
                "(import undetermined_head into the non-AttnComm actor); hybrid attn checkpoints use model_dir only."
            )
        _tm = str(getattr(all_args, "undet_v2_latent_train_mode", "finetune_all"))
        if _tm == "motion_only" and (not _lat_dir or not str(_lat_dir).strip()):
            raise ValueError(
                "--undet_v2_latent_train_mode motion_only requires --undet_v2_target_latent_model_dir "
                "(pretrained head path)"
            )
    if _lat_v3_dir and str(_lat_v3_dir).strip():
        _lat_v3_s = str(_lat_v3_dir)
        _head_arch = str(getattr(all_args, "undet_v3_head_arch", "global_rank_compat"))
        if ("decoupled_rank" in _lat_v3_s or "decoupled_equal" in _lat_v3_s) and _head_arch == "global_rank_compat":
            setattr(all_args, "undet_v3_head_arch", "decoupled_rank_compat")
            print("[train] auto-set undet_v3_head_arch=decoupled_rank_compat from latent checkpoint path")
        # Optional force-align for legacy runs; default keeps MAPPO #agents free to vary.
        _ckpt_agents = int(getattr(all_args, "undet_v3_latent_dataset_num_agents", 10))
        if bool(getattr(all_args, "undet_v3_latent_force_dataset_num_agents", False)):
            if int(all_args.num_agents) != _ckpt_agents:
                print(
                    f"[train] num_agents {int(all_args.num_agents)} -> {_ckpt_agents} "
                    f"(forced by --undet_v3_latent_force_dataset_num_agents)"
                )
                all_args.num_agents = _ckpt_agents
        else:
            print(
                f"[train] keep num_agents={int(all_args.num_agents)} "
                f"(v3 latent dataset_num_agents={_ckpt_agents}, no force)"
            )
        if not getattr(all_args, "enable_undetermined_goal", False):
            raise ValueError("--undet_v3_target_latent_model_dir requires --enable_undetermined_goal")
        if not getattr(all_args, "enable_undetermined_goal_v3", False):
            raise ValueError("--undet_v3_target_latent_model_dir requires --enable_undetermined_goal_v3")
        if not getattr(all_args, "use_attn_comm_actor", False):
            raise ValueError("--undet_v3_target_latent_model_dir requires --use_attn_comm_actor")
        if str(getattr(all_args, "architecture_mode", "default")) != "attn_undetermined_goal":
            raise ValueError(
                "--undet_v3_target_latent_model_dir requires --architecture_mode attn_undetermined_goal"
            )
        _tm3 = str(getattr(all_args, "undet_v3_latent_train_mode", "finetune_all"))
        if _tm3 == "motion_only" and (not _lat_v3_dir or not str(_lat_v3_dir).strip()):
            raise ValueError(
                "--undet_v3_latent_train_mode motion_only requires --undet_v3_target_latent_model_dir"
            )

    if getattr(all_args, "enable_undetermined_goal", False):
        all_args.enable_dynamic_goal_assignment = False
        k = int(all_args.num_agents)
        if getattr(all_args, "enable_undetermined_goal_v3", False):
            if not getattr(all_args, "use_attn_comm_actor", False):
                raise ValueError("--enable_undetermined_goal_v3 requires --use_attn_comm_actor")
            if str(getattr(all_args, "architecture_mode", "default")) != "attn_undetermined_goal":
                raise ValueError(
                    "--enable_undetermined_goal_v3 requires --architecture_mode attn_undetermined_goal"
                )
            if str(getattr(all_args, "undet_v2_head_arch", "dot_product")) == "pair_mlp":
                raise ValueError("--enable_undetermined_goal_v3 requires --undet_v2_head_arch dot_product")
            all_args.enable_undetermined_goal_v2 = False
            p_v3 = max(0, int(getattr(all_args, "undetermined_v3_comm_ally_slots", 6)))
            h_v3 = max(0, int(getattr(all_args, "undetermined_v3_comm_human_slots", 4)))
            all_args.attn_comm_ally_slots = p_v3
            all_args.attn_comm_human_slots = h_v3
            m = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
            md = int(getattr(all_args, "attn_comm_message_dim", 16))
            all_args.robot_obs_dim = compute_undetermined_v3_robot_obs_dim(m, p_v3, h_v3, md)
            apply_undetermined_v2_reward_floors(all_args)
            print(
                f"[train] undetermined goal v3: robot_obs_dim={all_args.robot_obs_dim} (+2 px,py; +1 prev_tid_norm), K={k}, "
                f"M={m}, comm_ally_slots={p_v3}, comm_human_slots={h_v3} (agent-count decoupled), "
                f"hungarian_div={getattr(all_args, 'undetermined_v2_hungarian_team_divisor', 8.0)}, "
                f"v3_target_kl_coef={float(getattr(all_args, 'undetermined_v3_target_kl_coef', 0.0))}"
            )
            print(
                f"[train] undetermined v3 uses v2-style S_L reward: dense_scale={float(getattr(all_args, 'undetermined_v2_sl_dense_scale', 0.0))}, "
                f"delta_scale={float(getattr(all_args, 'undetermined_v2_sl_delta_scale', 0.0))}, "
                f"success_scale={float(getattr(all_args, 'undetermined_v2_sl_success_scale', 0.0))}, "
                f"thr={float(getattr(all_args, 'undetermined_v2_sl_success_threshold', 0.97))}"
            )
        elif getattr(all_args, "enable_undetermined_goal_v2", False):
            m = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
            all_args.robot_obs_dim = compute_undetermined_v2_robot_obs_dim(m)
            apply_undetermined_v2_reward_floors(all_args)
            print(
                f"[train] undetermined goal v2: robot_obs_dim={all_args.robot_obs_dim} (+2 px,py), "
                f"K={k}, v2_slots={m}, hungarian_div={getattr(all_args, 'undetermined_v2_hungarian_team_divisor', 8.0)}"
            )
            if getattr(all_args, "enable_undetermined_v2_exchange", False):
                _xr = getattr(all_args, "undetermined_v2_exchange_radius", None)
                _xr_s = float(_xr) if _xr is not None else float(getattr(all_args, "undetermined_comm_radius", 6.0))
                _ex_crit = str(getattr(all_args, "undetermined_v2_exchange_accept_criterion", "fleet_m"))
                print(
                    f"[train] undetermined v2_exchange (accept={_ex_crit}; "
                    f"fleet_m uses fleet M=max_k dist2goal; cone_mutual_greedy_m uses cones+mutual+greedy M): "
                    f"radius={_xr_s}, "
                    f"min_gain={float(getattr(all_args, 'undetermined_v2_exchange_min_gain', 0.05))}, "
                    f"max_pairs/step={int(getattr(all_args, 'undetermined_v2_exchange_max_pairs_per_step', 1))}, "
                    f"ignore_pending={bool(getattr(all_args, 'undetermined_v2_exchange_ignore_pending', False))}"
                )
                print(
                    f"[train] v2_exchange reward shaping: M_drop_scale={float(getattr(all_args, 'undetermined_v2_exchange_bottleneck_shaping_scale', 0.0))}, "
                    f"S_drop_scale={float(getattr(all_args, 'undetermined_v2_exchange_team_dist_shaping_scale', 0.0))}, "
                    f"swap_S_bonus_scale={float(getattr(all_args, 'undetermined_v2_exchange_swap_bonus_scale', 0.0))}, "
                    f"div={float(getattr(all_args, 'undetermined_v2_exchange_shaping_team_divisor', 8.0))}"
                )
            print(
                f"[train] undetermined v2 type-2 (S_L) reward: dense_scale={float(getattr(all_args, 'undetermined_v2_sl_dense_scale', 0.0))}, "
                f"delta_scale={float(getattr(all_args, 'undetermined_v2_sl_delta_scale', 0.0))}, "
                f"success_scale={float(getattr(all_args, 'undetermined_v2_sl_success_scale', 0.0))}, "
                f"thr={float(getattr(all_args, 'undetermined_v2_sl_success_threshold', 0.97))}, "
                f"type2_pattern_first={bool(getattr(all_args, 'undetermined_v2_type2_pattern_first', False))}, "
                f"type2_formation_efficiency={bool(getattr(all_args, 'undetermined_v2_type2_formation_efficiency', False))}, "
                f"sl_post_lit={float(getattr(all_args, 'undetermined_v2_sl_post_success_literal_scale', 1.0))}, "
                f"sl_post_shape={float(getattr(all_args, 'undetermined_v2_sl_post_success_sl_shaping_scale', 1.0))}, "
                f"sl_succ_cross_only={bool(getattr(all_args, 'undetermined_v2_sl_success_only_on_crossing', False))}, "
                f"sl_succ_sustain={float(getattr(all_args, 'undetermined_v2_sl_success_sustain_frac', 0.0))}, "
                f"sl_pre_step={float(getattr(all_args, 'undetermined_v2_sl_pre_success_step_penalty', 0.0))}, "
                f"sl_pre_trav={float(getattr(all_args, 'undetermined_v2_sl_pre_success_travel_penalty', 0.0))}"
            )
            if _lat_dir and str(_lat_dir).strip():
                print(
                    f"[train] undet_v2_target_latent: train_mode={getattr(all_args, 'undet_v2_latent_train_mode', 'finetune_all')}, "
                    f"head_aux_coef={float(getattr(all_args, 'undet_v2_target_head_aux_coef', 0.0))}"
                )
        else:
            all_args.robot_obs_dim = compute_undetermined_robot_obs_dim(k)
            apply_undetermined_reward_floors(all_args)
            print(f"[train] undetermined goal mode: robot_obs_dim={all_args.robot_obs_dim} (+2 px,py), K={k}")
        print(
            f"[train] undetermined reward floors: nd_discount_avoid={getattr(all_args, 'nd_discount_avoid', 0)}, "
            f"nd_discount_nav={all_args.nd_discount_nav}, "
            f"nd_discount_goal={all_args.nd_discount_goal}, "
            f"nd_goal_progress_coef={all_args.nd_goal_progress_coef}, "
            f"nd_goal_terminal_reward={all_args.nd_goal_terminal_reward}, "
            f"nd_goal_leave_penalty={all_args.nd_goal_leave_penalty}, "
            f"undetermined_hungarian_reward_scale={all_args.undetermined_hungarian_reward_scale}, "
            f"undetermined_goal_distance_penalty_scale={all_args.undetermined_goal_distance_penalty_scale}, "
            f"undetermined_far_goal_progress_boost={all_args.undetermined_far_goal_progress_boost}"
        )

    if getattr(all_args, "use_attn_comm_actor", False):
        if getattr(all_args, "enable_dynamic_goal_assignment", False):
            raise ValueError(
                "use_attn_comm_actor requires fixed (pre-assigned) targets; "
                "disable --enable_dynamic_goal_assignment."
            )
        if getattr(all_args, "enable_undetermined_goal", False) and str(
            getattr(all_args, "architecture_mode", "default")
        ) != "attn_undetermined_goal":
            raise ValueError(
                "use_attn_comm_actor is incompatible with --enable_undetermined_goal unless "
                "--architecture_mode attn_undetermined_goal (hybrid ConsMAC observation pack)."
            )
        resolve_attn_comm_args(all_args)
        if str(getattr(all_args, "architecture_mode", "")) == "attn_undetermined_goal" and (
            getattr(all_args, "enable_undetermined_goal_v2", False)
            or getattr(all_args, "enable_undetermined_goal_v3", False)
        ):
            m = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
            if getattr(all_args, "enable_undetermined_goal_v3", False):
                all_args.robot_obs_dim = compute_undetermined_v3_robot_obs_dim(
                    m,
                    int(all_args.attn_comm_ally_slots),
                    int(all_args.attn_comm_human_slots),
                    int(getattr(all_args, "attn_comm_message_dim", 16)),
                )
            else:
                all_args.robot_obs_dim = compute_undetermined_v2_attn_hybrid_robot_obs_dim(
                    m,
                    int(all_args.attn_comm_ally_slots),
                    int(all_args.attn_comm_human_slots),
                    int(getattr(all_args, "attn_comm_message_dim", 16)),
                )
            _tl = compute_attn_comm_tail_dim(
                int(all_args.attn_comm_ally_slots),
                int(all_args.attn_comm_human_slots),
                int(getattr(all_args, "attn_comm_message_dim", 16)),
            )
            _mode = "v3" if getattr(all_args, "enable_undetermined_goal_v3", False) else "v2"
            print(
                f"[train] architecture attn_undetermined_goal ({_mode}): hybrid robot_obs_dim={all_args.robot_obs_dim} (+2), "
                f"undetermined_v2_core={compute_undetermined_v2_robot_obs_dim(m)}, consmac_tail={_tl}"
            )
        else:
            all_args.robot_obs_dim = compute_attn_comm_robot_obs_dim(
                int(all_args.attn_comm_ally_slots),
                int(all_args.attn_comm_human_slots),
                message_dim=int(getattr(all_args, "attn_comm_message_dim", 16)),
            )
        resolve_attn_comm_ppo_batch_args(all_args)

    if getattr(all_args, "enable_dynamic_goal_assignment", False):
        k = int(all_args.num_agents)
        all_args.dynamic_obs_pack_version = getattr(all_args, "dynamic_obs_pack_version", "slots")
        all_args.use_neighbor_attn_lstm_actor = not getattr(
            all_args, "disable_neighbor_attn_lstm_actor", False
        )
        all_args.robot_obs_dim = compute_dynamic_robot_obs_dim(
            k,
            int(all_args.dynamic_target_slot_count),
            use_neighbor_attn_lstm_actor=bool(all_args.use_neighbor_attn_lstm_actor),
            actor_neighbor_n=int(all_args.actor_neighbor_n),
        )
        if all_args.use_neighbor_attn_lstm_actor:
            print(
                f"[train] actor neighbor self-attn+LSTM: P={min(int(all_args.actor_neighbor_n), max(0, k - 1))} "
                f"nearest teammates (slot_m={min(max(int(all_args.dynamic_target_slot_count), 1), k)})"
            )
    else:
        all_args.use_neighbor_attn_lstm_actor = False

    return all_args


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    all_args.num_humans = 2
    all_args.num_attention_agents = 10
    all_args.n_rollout_threads = 40
    all_args.episode_length = 400
    all_args.num_env_steps = all_args.n_rollout_threads * all_args.episode_length * 1200
    # PPO: mini_batch_size = (n_rollout_threads * episode_length * num_agents) / num_mini_batch.
    # Too few mini-batches => huge CUDA batches and OOM (e.g. 10 -> ~80k samples/step on 200x400x10).
    # Slightly fewer splits than 400 => larger per-GPU minibatch (stabler grads for formation shaping).
    all_args.num_mini_batch = 400
    all_args.save_interval = 1
    all_args.log_interval = 1
    all_args.model_dir = None
    # all_args.model_dir = '/home/yao/mappo/mappo1.8.4.2/results/train/run12/models'

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError


    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (Path(os.path.dirname(os.path.abspath(__file__)) + "/results" + "/train")) # split表示拆分路径，索引为0表示返回拆分后的路径
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    print("run_dir",run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    _notes_path = run_dir / "run_flags.txt"
    _asm = getattr(all_args, "agent_state_mode", "all")
    _dyn = bool(getattr(all_args, "enable_dynamic_goal_assignment", False))
    _und = bool(getattr(all_args, "enable_undetermined_goal", False))
    _ir = bool(getattr(all_args, "randomize_robot_initial_positions", False))
    _notes = (
        f"agent_state_mode: {_asm}\n"
        f"architecture_mode: {str(getattr(all_args, 'architecture_mode', 'default'))}\n"
        f"dynamic_target: {_dyn}\n"
        f"undetermined_goal: {_und}\n"
        f"initial_randomize: {_ir}\n"
        f"robot_initial_spawn_mode: {str(getattr(all_args, 'robot_initial_spawn_mode', 'random_box'))}\n"
        f"robot_init_cluster_radius_mode: {str(getattr(all_args, 'robot_init_cluster_radius_mode', 'comm'))}\n"
        f"undetermined_v2_exchange_accept_criterion: "
        f"{str(getattr(all_args, 'undetermined_v2_exchange_accept_criterion', 'fleet_m'))}\n"
        f"num_agents: {int(all_args.num_agents)}\n"
    )
    if _asm == "nearest_n_radius":
        _nn = int(getattr(all_args, "neighbor_n", 5))
        _nr = float(getattr(all_args, "neighbor_radius", 5.0))
        _notes += f"neighbor_n: {_nn}\nneighbor_radius: {_nr}\n"
    if _dyn:
        _notes += (
            f"dynamic_target_slot_count: {int(getattr(all_args, 'dynamic_target_slot_count', 0))}\n"
            f"dynamic_target_vis_radius: {float(getattr(all_args, 'dynamic_target_vis_radius', 0.0))}\n"
            f"use_neighbor_attn_lstm_actor: {bool(getattr(all_args, 'use_neighbor_attn_lstm_actor', False))}\n"
            f"actor_neighbor_n: {int(getattr(all_args, 'actor_neighbor_n', 0))}\n"
            f"dynamic_discount_formation: {float(getattr(all_args, 'dynamic_discount_formation', 0.0))}\n"
            f"formation_time_weight: {float(getattr(all_args, 'formation_time_weight_start', 1.0))}"
            f" -> {float(getattr(all_args, 'formation_time_weight_end', 1.0))}\n"
            f"formation_time_weight_decay_horizon: {float(getattr(all_args, 'formation_time_weight_decay_horizon', 1.0))}\n"
            f"dynamic_arrival_require_outside_entry: {int(getattr(all_args, 'dynamic_arrival_require_outside_entry', 1))}\n"
            f"dynamic_crowding_dist/penalty: {float(getattr(all_args, 'dynamic_crowding_dist', 0.0))}"
            f" / {float(getattr(all_args, 'dynamic_crowding_penalty_scale', 0.0))}\n"
            f"dynamic_cluster_same_target_boost: {float(getattr(all_args, 'dynamic_cluster_same_target_boost', 1.0))}\n"
            f"dynamic_explore_undervisible_scale: {float(getattr(all_args, 'dynamic_explore_undervisible_scale', 0.0))}\n"
            f"dynamic_shaping_commit_rvis_mult: {float(getattr(all_args, 'dynamic_shaping_commit_rvis_mult', 0.0))}\n"
            f"dynamic_reciprocal_swap_reward_scale: {float(getattr(all_args, 'dynamic_reciprocal_swap_reward_scale', 0.0))}\n"
            f"dynamic_goal_contention_penalty_scale: {float(getattr(all_args, 'dynamic_goal_contention_penalty_scale', 0.0))}\n"
            f"dynamic_claimed_target_penalty_scale: {float(getattr(all_args, 'dynamic_claimed_target_penalty_scale', 0.0))}\n"
            f"dynamic_low_density_explore_scale: {float(getattr(all_args, 'dynamic_low_density_explore_scale', 0.0))}\n"
            f"use_centralized_V: {bool(getattr(all_args, 'use_centralized_V', False))}\n"
            f"dynamic_ctde_remaining_target_shaping_scale: {float(getattr(all_args, 'dynamic_ctde_remaining_target_shaping_scale', 0.0))}\n"
        )
    if _und:
        _notes += (
            f"robot_obs_dim: {int(getattr(all_args, 'robot_obs_dim', 0))}\n"
            f"undetermined_obs_goal_radius: {float(getattr(all_args, 'undetermined_obs_goal_radius', 5.0))}\n"
            f"undetermined_comm_radius: {float(getattr(all_args, 'undetermined_comm_radius', 6.0))}\n"
            f"undetermined_hungarian_reward_scale: {float(getattr(all_args, 'undetermined_hungarian_reward_scale', 0.10))}\n"
            f"undetermined_target_embed_dim: {int(getattr(all_args, 'undetermined_target_embed_dim', 32))}\n"
            f"undetermined_max_auction_rounds: {int(getattr(all_args, 'undetermined_max_auction_rounds', 8))}\n"
            f"undetermined_far_goal_progress_dist_thresh: {float(getattr(all_args, 'undetermined_far_goal_progress_dist_thresh', 4.0))}\n"
            f"undetermined_far_goal_progress_boost: {float(getattr(all_args, 'undetermined_far_goal_progress_boost', 1.6))}\n"
            f"undetermined_goal_distance_penalty_scale: {float(getattr(all_args, 'undetermined_goal_distance_penalty_scale', 0.004))}\n"
            f"use_centralized_V: {bool(getattr(all_args, 'use_centralized_V', False))}\n"
            f"formation_time_weight: {float(getattr(all_args, 'formation_time_weight_start', 1.0))}"
            f" -> {float(getattr(all_args, 'formation_time_weight_end', 1.0))}\n"
            f"formation_time_weight_decay_horizon: {float(getattr(all_args, 'formation_time_weight_decay_horizon', 1.0))}\n"
            f"nd_discount_formation: {float(getattr(all_args, 'nd_discount_formation', 0.0))}\n"
            f"nd_discount_avoid: {float(getattr(all_args, 'nd_discount_avoid', 50.0))}\n"
            f"nd_discount_nav: {float(getattr(all_args, 'nd_discount_nav', 20.0))}\n"
            f"nd_discount_goal: {float(getattr(all_args, 'nd_discount_goal', 200.0))}\n"
            f"nd_goal_progress_coef: {float(getattr(all_args, 'nd_goal_progress_coef', 5.0))}\n"
            f"nd_arrival_reward: {float(getattr(all_args, 'nd_arrival_reward', 0.0))}\n"
            f"nd_goal_terminal_reward: {float(getattr(all_args, 'nd_goal_terminal_reward', 0.0))}\n"
            f"nd_proximity_reward_scale: {float(getattr(all_args, 'nd_proximity_reward_scale', 0.0))}\n"
            f"nd_heading_reward_scale: {float(getattr(all_args, 'nd_heading_reward_scale', 0.0))}\n"
        )
    if bool(getattr(all_args, "use_attn_comm_actor", False)):
        _notes += (
            f"use_attn_comm_actor: True\n"
            f"attn_comm_radius: {float(getattr(all_args, 'attn_comm_radius', 0.0))}\n"
            f"attn_comm_ally_slots: {int(getattr(all_args, 'attn_comm_ally_slots', 0))}\n"
            f"attn_comm_human_slots: {int(getattr(all_args, 'attn_comm_human_slots', 0))}\n"
            f"attn_comm_max_ppo_samples_per_gpu: {int(getattr(all_args, 'attn_comm_max_ppo_samples_per_gpu', 0))}\n"
        )
    _lat_note = getattr(all_args, "undet_v2_target_latent_model_dir", None)
    if _lat_note and str(_lat_note).strip():
        _notes += f"undet_v2_target_latent_model_dir: {_lat_note}\n"
        _notes += f"undet_v2_latent_train_mode: {getattr(all_args, 'undet_v2_latent_train_mode', 'finetune_all')}\n"
        _notes += f"undet_v2_target_head_aux_coef: {float(getattr(all_args, 'undet_v2_target_head_aux_coef', 0.0))}\n"
    _lat_v3_note = getattr(all_args, "undet_v3_target_latent_model_dir", None)
    if _lat_v3_note and str(_lat_v3_note).strip():
        _notes += f"undet_v3_target_latent_model_dir: {_lat_v3_note}\n"
        _notes += f"undet_v3_latent_train_mode: {getattr(all_args, 'undet_v3_latent_train_mode', 'finetune_all')}\n"
    _notes += f"num_mini_batch: {int(all_args.num_mini_batch)}\n"
    with open(_notes_path, "w", encoding="utf-8") as _nf:
        _nf.write(_notes)

    setproctitle.setproctitle("@" + str(all_args.user_name))
    # for i in range(5):
    # seed
    # all_args.num_humans = 3 + i
    torch.manual_seed(all_args.seed*200)
    torch.cuda.manual_seed_all(all_args.seed*200)
    np.random.seed(all_args.seed*200)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experimentsFalse
    if True:
        from runner.shared.env_runner import EnvRunner as Runner

        runner = Runner(config)
        runner.run()
    else:
        from runner.separated.env_runner import EnvRunner as Runner

        runner = Runner(config)
        runner.run(all_args)



    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()

    print('model save in:',run_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
