# !/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

# import rvo2
from config.config import get_config          
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():

            from envs.env_discrete import DiscreteActionEnv
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
            env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])    #单线程
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])   #多线程


def parser_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=15, help="number of players")
    parser.add_argument("--random_act_prob", type=int, default=0, help="the probability of robot to choice random action")
    parser.add_argument('--config_path', type=str, default='reward.yaml',
                        help='Path to reward function config file')

    all_args = parser.parse_known_args(args)[0]

    # DDP: get local_rank from environment variable (set by torchrun)
    if 'LOCAL_RANK' in os.environ:
        all_args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        all_args.local_rank = 0

    return all_args


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    all_args.num_humans = 2
    all_args.num_attention_agents = 10
    all_args.n_rollout_threads = 100
    all_args.episode_length = 400
    all_args.num_env_steps = all_args.n_rollout_threads * all_args.episode_length * 800
    all_args.num_mini_batch = 8
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


    # cuda and DDP setup
    if all_args.cuda and torch.cuda.is_available():
        if all_args.use_ddp:
            # DDP mode: initialize distributed training
            dist.init_process_group(backend='nccl')
            local_rank = all_args.local_rank
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda:{}'.format(local_rank))
            world_size = dist.get_world_size()
            print("DDP mode: using GPU {} (world_size={})".format(local_rank, world_size))
            # DDP: 每卡保持100环境，通过减少episodes实现加速，总采样量不变
            # 单卡: 100环境 × 400步 × 1200 episodes = 48M 总步数，时间T
            # DDP双卡: 每卡100环境 × 400步 × 600 episodes = 48M 总步数(两卡合计)，时间T/2
            # DDP梯度同步使两卡共享梯度，等效于每步batch翻倍，模型质量不受影响
            all_args.num_env_steps = all_args.n_rollout_threads * all_args.episode_length * 600
            print("DDP: n_rollout_threads={}, episodes=600, num_env_steps={} (total across {} GPUs = {})".format(
                all_args.n_rollout_threads, all_args.num_env_steps, world_size,
                all_args.num_env_steps * world_size))
        else:
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

    # run dir - only rank 0 creates directory in DDP mode
    run_dir = (Path(os.path.dirname(os.path.abspath(__file__)) + "/results" + "/train"))

    if all_args.use_ddp:
        # DDP mode: only rank 0 determines and creates run_dir
        if all_args.local_rank == 0:
            if not run_dir.exists():
                os.makedirs(str(run_dir))
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
            if not run_dir.exists():
                os.makedirs(str(run_dir))
            # Broadcast run_dir string to other ranks
            run_dir_str = str(run_dir)
            run_dir_list = [run_dir_str]
            dist.broadcast_object_list(run_dir_list, src=0)
        else:
            # Non-rank-0 processes receive run_dir from rank 0
            run_dir_list = [""]
            dist.broadcast_object_list(run_dir_list, src=0)
            run_dir = Path(run_dir_list[0])
        print("run_dir", run_dir)
    else:
        # Single GPU mode: original logic
        if not run_dir.exists():
            os.makedirs(str(run_dir))
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
        print("run_dir", run_dir)
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle("@" + str(all_args.user_name))
    # for i in range(5):
    # seed
    # all_args.num_humans = 3 + i
    # DDP: 每个进程使用不同的seed，确保环境多样性
    seed_offset = all_args.local_rank if all_args.use_ddp else 0
    torch.manual_seed((all_args.seed + seed_offset) * 200)
    torch.cuda.manual_seed_all((all_args.seed + seed_offset) * 200)
    np.random.seed((all_args.seed + seed_offset) * 200)

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
        "use_ddp": all_args.use_ddp,
        "local_rank": all_args.local_rank if all_args.use_ddp else 0,
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

    # Only rank 0 saves summary and prints
    if not all_args.use_ddp or all_args.local_rank == 0:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()
        print('model save in:',run_dir)

    # Clean up DDP
    if all_args.use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
