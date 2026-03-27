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
from config.config import get_config          
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

    all_args = parser.parse_known_args(args)[0]

    # Match swarm size to font-pattern length bucket (config --train_font_pattern_length).
    _pat_len = int(getattr(all_args, "train_font_pattern_length", 10))
    _prev_n = int(all_args.num_agents)
    if _prev_n != _pat_len:
        print(f"[train] num_agents {_prev_n} -> {_pat_len} (aligned to train_font_pattern_length)")
    all_args.num_agents = _pat_len

    if getattr(all_args, "enable_dynamic_goal_assignment", False):
        k = int(all_args.num_agents)
        all_args.robot_obs_dim = 7 + 3 * k + max(0, k - 1)

    return all_args


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    all_args.num_humans = 2
    all_args.num_attention_agents = 10
    all_args.n_rollout_threads = 200
    all_args.episode_length = 400
    all_args.num_env_steps = all_args.n_rollout_threads * all_args.episode_length * 1200
    all_args.num_mini_batch = 10
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
    _ir = bool(getattr(all_args, "randomize_robot_initial_positions", False))
    _notes = (
        f"agent_state_mode: {_asm}\n"
        f"dynamic_target: {_dyn}\n"
        f"initial_randomize: {_ir}\n"
        f"num_agents: {int(all_args.num_agents)}\n"
    )
    if _asm == "nearest_n_radius":
        _nn = int(getattr(all_args, "neighbor_n", 5))
        _nr = float(getattr(all_args, "neighbor_radius", 5.0))
        _notes += f"neighbor_n: {_nn}\nneighbor_radius: {_nr}\n"
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
