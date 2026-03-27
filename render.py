#!/usr/bin/env python
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import setproctitle
import numpy as np
from pathlib import Path

import torch

from config.config import get_config

from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def apply_dynamic_goal_obs_dim(all_args):
    """Match DiscreteActionEnv / training: dynamic mode needs larger robot_obs_dim."""
    if getattr(all_args, "enable_dynamic_goal_assignment", False):
        k = int(all_args.num_agents)
        all_args.robot_obs_dim = 7 + 3 * k + max(0, k - 1)


def align_render_args_from_actor_checkpoint(all_args):
    """
    Read results/.../models/actor.pt and set enable_dynamic_goal_assignment,
    num_agents, and robot_obs_dim so the policy matches the checkpoint.
    """
    md = Path(str(getattr(all_args, "model_dir", "") or ""))
    actor_path = md / "actor.pt"
    if not actor_path.is_file():
        print(f"[render] auto_align: no {actor_path}, skip")
        return
    try:
        sd = torch.load(actor_path, map_location="cpu")
    except Exception as e:
        print(f"[render] auto_align: failed to load actor.pt: {e}")
        return
    feat_key = "base_robot.feature_norm.weight"
    if feat_key not in sd:
        print("[render] auto_align: unexpected actor layout, skip")
        return
    D = int(sd[feat_key].shape[0])
    third_bias_key = "act.action_outs.2.linear.bias"
    if third_bias_key in sd:
        all_args.enable_dynamic_goal_assignment = True
        K = int(sd[third_bias_key].shape[0])
        all_args.num_agents = K
        all_args.robot_obs_dim = 7 + 3 * K + max(0, K - 1)
        if all_args.robot_obs_dim + 2 != D:
            print(
                f"[render] auto_align WARN: ckpt feat dim {D} vs "
                f"robot_obs_dim+2={all_args.robot_obs_dim + 2} (num_agents={K})"
            )
        print(
            f"[render] auto_align: dynamic goals, num_agents={K}, "
            f"robot_obs_dim={all_args.robot_obs_dim} (actor input dim {D})"
        )
    else:
        all_args.enable_dynamic_goal_assignment = False
        all_args.robot_obs_dim = max(7, D - 2)
        print(
            f"[render] auto_align: non-dynamic, robot_obs_dim={all_args.robot_obs_dim} "
            f"(actor input dim {D})"
        )


def _warn_checkpoint_dynamic_goal_head_mismatch(all_args):
    if not getattr(all_args, "enable_dynamic_goal_assignment", False):
        return
    md = Path(str(getattr(all_args, "model_dir", "") or ""))
    actor_path = md / "actor.pt"
    if not actor_path.is_file():
        return
    try:
        sd = torch.load(actor_path, map_location="cpu")
    except Exception:
        return
    key = "act.action_outs.2.linear.bias"
    if key not in sd:
        return
    k_ckpt = int(sd[key].shape[0])
    n = int(all_args.num_agents)
    if k_ckpt != n:
        print(
            f"[render] WARN: checkpoint third-action head K={k_ckpt} != num_agents={n} "
            f"after nearest_n_radius pattern align; policy load may fail unless K matches."
        )


def align_num_agents_to_font_pattern_if_nearest_n_radius(all_args):
    """
    When critic uses nearest_n_radius slots, set num_agents to the selected render
    pattern's coordinate count (same selection as EnvCore), independent of dynamic_target.
    Runs after checkpoint auto-align so pattern length overrides ckpt K for env sizing.
    """
    if getattr(all_args, "agent_state_mode", "all") != "nearest_n_radius":
        return
    from envs.env_core import select_font_pattern_targets_for_args

    saved_rank = int(getattr(all_args, "env_rank", 0))
    all_args.env_rank = 0
    try:
        name, coords = select_font_pattern_targets_for_args(all_args, log_selection=False)
    finally:
        all_args.env_rank = saved_rank

    n = len(coords)
    prev = int(all_args.num_agents)
    if n != prev:
        print(
            f"[render] nearest_n_radius: num_agents {prev} -> {n} "
            f"(pattern '{name}', {n} coordinates)"
        )
    else:
        print(
            f"[render] nearest_n_radius: num_agents={n} (pattern '{name}')"
        )
    all_args.num_agents = n
    apply_dynamic_goal_obs_dim(all_args)
    _warn_checkpoint_dynamic_goal_head_mismatch(all_args)


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from envs.env_discrete import DiscreteActionEnv
            all_args.env_rank = rank
            env = DiscreteActionEnv(all_args)

            env.seed(all_args.seed + rank * 1000 )
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)], all_args)
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def parser_args(args, parser):
    parser.add_argument('--num_agents', type=int,default=10, help="number of players")
    parser.add_argument("--random_act_prob", type=int, default=0, help="the probability of robot to choice random action")
    parser.add_argument(
        "--no_render_auto_align_checkpoint",
        action="store_true",
        default=False,
        help="Disable reading actor.pt to match dynamic goals / num_agents / robot_obs_dim (auto-align is on by default).",
    )

    all_args = parser.parse_known_args(args)[0]

    apply_dynamic_goal_obs_dim(all_args)

    return all_args


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    all_args.use_render = True
    all_args.model_dir = '/home/wangdx_lab/cse12211818/MAPPO/results/train/run4/models'
    all_args.n_rollout_threads = 1
    all_args.episode_length = 500
    all_args.visualize = False
    all_args.render_episodes = 100
    all_args.num_attention_agents = 5
    all_args.num_humans = 2
    all_args.method = 'ppo'

    if all_args.method == 'ppo':
        if all_args.algorithm_name == "rmappo":
            print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
            all_args.use_recurrent_policy = True
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "mappo":
            print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
            all_args.use_recurrent_policy = False 
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "ippo":
            print("u are choosing to use ippo, we set use_centralized_V to be False.")
            all_args.use_centralized_V = False
        else:
            raise NotImplementedError
    elif all_args.method == 'orca':
        print('u chose orca method!')
    elif all_args.method == 'apf':
        print('u chose ipf method!')

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads==1, ("only support to use 1 env to render.")

    # After model_dir is set: match checkpoint (dynamic target head + obs dim) unless disabled.
    if not getattr(all_args, "no_render_auto_align_checkpoint", False):
        align_render_args_from_actor_checkpoint(all_args)
    apply_dynamic_goal_obs_dim(all_args)
    align_num_agents_to_font_pattern_if_nearest_n_radius(all_args)

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
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results" + "/render")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    print(f'================ Results Saved in {curr_run} ================\n')
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle("@" + str(all_args.user_name))     #进程名称

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    if True:
        from runner.shared.env_runner import EnvRunner as Runner

        runner = Runner(config)
        runner.render()
    else:
        from runner.separated.env_runner import EnvRunner as Runner

        runner = Runner(config)
        runner.render(mode='vedio', visualize=all_args.visualize, method=all_args.method)



    
    
    # post process
    # envs.close()
    print('accomplish test,test episode:',all_args.render_episodes)

if __name__ == "__main__":
    main(sys.argv[1:])
