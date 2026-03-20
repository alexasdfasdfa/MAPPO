#!/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path

import torch

from config.config import get_config

from envs.env_wrappers import DummyVecEnv, SubprocVecEnv

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
    
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    all_args.use_render = True
    all_args.model_dir = '/home/wangdx_lab/cse12211818/mappo1.8.4.2/results/train/run116/models'
    all_args.n_rollout_threads = 1
    all_args.episode_length = 350
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
