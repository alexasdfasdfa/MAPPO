#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAPPO模型评估脚本
评估训练好的模型，生成评估指标和可视化图表
"""

import sys
import os
import numpy as np
from pathlib import Path
import torch
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from config.config import get_config
from envs.env_wrappers import DummyVecEnv
from envs.utils.utils import reach_goal


def parser_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=15)
    parser.add_argument("--random_act_prob", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="reward.yaml")
    return parser.parse_known_args(args)[0]


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)], all_args)


def load_training_logs(log_dir):
    """加载训练日志 - 优先从summary.json，否则从TensorBoard事件文件"""
    logs = {}
    summary_path = os.path.join(log_dir, "summary.json")

    if os.path.exists(summary_path):
        # 使用summary.json
        with open(summary_path, "r") as f:
            logs["summary"] = json.load(f)
    else:
        # 从TensorBoard事件文件加载
        print("summary.json not found, loading from TensorBoard event files...")
        summary = {}

        # TensorBoard日志目录中的指标子目录
        metric_dirs = [
            "average_episode_rewards", "policy_loss", "value_loss",
            "dist_entropy", "actor_grad_norm", "critic_grad_norm"
        ]

        try:
            from tensorboard.backend.event_processing import event_accumulator
        except ImportError:
            print("Warning: tensorboard not installed, cannot load event files")
            return logs

        for metric_name in metric_dirs:
            metric_path = os.path.join(log_dir, metric_name)
            if not os.path.exists(metric_path):
                continue

            # 查找事件文件
            event_files = []
            for root, dirs, files in os.walk(metric_path):
                for f in files:
                    if f.startswith("events.out.tfevents"):
                        event_files.append(os.path.join(root, f))

            if not event_files:
                continue

            # 读取事件文件
            data_points = []
            for event_file in sorted(event_files):
                try:
                    ea = event_accumulator.EventAccumulator(event_file)
                    ea.Reload()

                    # 获取标量数据
                    tags = ea.Tags().get("scalars", [])
                    for tag in tags:
                        for event in ea.Scalars(tag):
                            # 格式: [wall_time, step, value]
                            data_points.append([event.wall_time, event.step, event.value])
                except Exception as e:
                    print(f"Warning: Failed to load {event_file}: {e}")
                    continue

            if data_points:
                # 按step排序
                data_points.sort(key=lambda x: x[1])
                summary[metric_name] = data_points

        if summary:
            logs["summary"] = summary

    return logs


def plot_training_curves(logs, save_dir):
    """绘制训练曲线"""
    print("Generating training curves...")
    summary = logs.get("summary", {})

    if not summary:
        print("No training data available to plot training curves")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Training Metrics", fontsize=14)

    # 指标配置: (summary中的key, y轴标签, 颜色)
    # 兼容summary.json和TensorBoard两种格式
    metrics_config = [
        (["average_episode_rewards"], "Average Episode Reward", "b"),
        (["policy_loss"], "Policy Loss", "g"),
        (["value_loss"], "Value Loss", "m"),
        (["dist_entropy"], "Distribution Entropy", "c"),
        (["actor_grad_norm"], "Actor Grad Norm", "orange"),
        (["critic_grad_norm"], "Critic Grad Norm", "purple")
    ]

    for idx, (key_candidates, ylabel, color) in enumerate(metrics_config):
        ax = axes[idx // 3, idx % 3]

        # 查找匹配的key
        data_key = None
        for candidate in key_candidates:
            # 先尝试精确匹配
            if candidate in summary:
                data_key = candidate
                break
            # 再尝试包含匹配
            for k in summary.keys():
                if candidate in k:
                    data_key = k
                    break
            if data_key:
                break

        if data_key:
            data = summary[data_key]
            steps = [d[1] for d in data]
            values = [d[2] for d in data]
            ax.plot(steps, values, color=color, linewidth=1.5)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for\n{ylabel}", transform=ax.transAxes,
                   ha='center', va='center', fontsize=10)
            ax.set_title(ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}")


def plot_evaluation_results(results, save_dir):
    """绘制评估结果"""
    print("Generating evaluation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Evaluation Results", fontsize=14)

    # Episode Rewards
    ax = axes[0, 0]
    rewards = results["episode_rewards"]
    ax.hist(rewards, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(rewards), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(rewards):.2f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Episode Rewards Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode Lengths
    ax = axes[0, 1]
    lengths = results["episode_lengths"]
    ax.hist(lengths, bins=20, color="coral", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(lengths):.1f}")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Frequency")
    ax.set_title("Episode Lengths Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success Rate
    ax = axes[1, 0]
    success_rate = np.cumsum(results["successes"]) / (np.arange(len(results["successes"])) + 1)
    ax.plot(range(1, len(success_rate) + 1), success_rate, "b-", linewidth=2)
    ax.axhline(success_rate[-1], color="red", linestyle="--",
               label=f"Final: {success_rate[-1]*100:.1f}%")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Success Rate")
    ax.set_title("Success Rate Convergence")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = f"""
    ===== Evaluation Summary =====

    Total Episodes: {len(rewards)}

    Reward Statistics:
      - Mean: {np.mean(rewards):.4f}
      - Std: {np.std(rewards):.4f}
      - Max: {np.max(rewards):.4f}
      - Min: {np.min(rewards):.4f}

    Success Rate: {results["success_rate"]*100:.2f}%
    Collision Rate: {results["collision_rate"]*100:.2f}%
    Timeout Rate: {results["timeout_rate"]*100:.2f}%

    Avg Episode Length: {np.mean(lengths):.1f} steps
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluation_results.png"), dpi=150)
    plt.close()
    print(f"Evaluation plots saved to {save_dir}")


def evaluate_model(all_args, envs, policy, num_episodes=100):
    """评估模型"""
    print(f"Evaluating model for {num_episodes} episodes...")

    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "successes": [],
        "collisions": [],
        "timeouts": [],
        "agent_success_count": [],  # 每个episode中成功到达目标的智能体数
        "total_agents": all_args.num_agents  # 总智能体数
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for ep in range(num_episodes):
        obs = envs.reset()
        episode_reward = 0
        step = 0
        done = False
        collision = False
        success = False
        timeout = False
        agent_successes = 0  # 本episode中成功到达目标的智能体数（在结束时统计）

        rnn_states = np.zeros((1, all_args.num_agents, 1, all_args.hidden_size), dtype=np.float32)
        masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)

        while not done and step < all_args.episode_length:
            obs_raw = obs[0]
            robot_obs_dim = all_args.robot_obs_dim + 2
            human_obs_dim = all_args.human_obs_dim
            human_num = all_args.num_humans

            robot_obs = obs_raw[:, 0, :robot_obs_dim]
            human_obs = obs_raw[:, 1:1+human_num, :human_obs_dim] if human_num > 0 else np.zeros((all_args.num_agents, 0, human_obs_dim))

            with torch.no_grad():
                robot_obs_t = torch.as_tensor(robot_obs, dtype=torch.float32, device=device)
                human_obs_t = torch.as_tensor(human_obs, dtype=torch.float32, device=device)
                rnn_states_t = torch.as_tensor(rnn_states.reshape(-1, all_args.hidden_size), dtype=torch.float32, device=device)
                masks_t = torch.as_tensor(masks.reshape(-1, 1), dtype=torch.float32, device=device)

                actions, new_rnn_states = policy.act(robot_obs_t, human_obs_t, rnn_states_t, masks_t, deterministic=True)

                actions_np = actions.cpu().numpy()
                rnn_states = new_rnn_states.cpu().numpy().reshape(1, all_args.num_agents, 1, all_args.hidden_size)

            obs, rewards, dones, infos = envs.step(actions_np)
            episode_reward += np.sum(rewards[0])
            step += 1

            if hasattr(envs.env, "env") and hasattr(envs.env.env, "collision_flag"):
                if envs.env.env.collision_flag:
                    collision = True

            if step >= all_args.episode_length:
                timeout = True

            done = dones[0][0] if isinstance(dones[0], (list, np.ndarray)) else dones[0]

        # episode结束时统计每个智能体是否到达目标
        if hasattr(envs.env, "env") and hasattr(envs.env.env, "robots"):
            agent_successes = 0
            for robot in envs.env.env.robots:
                if reach_goal(robot):
                    agent_successes += 1
            # 所有智能体都到达才算episode成功
            success = (agent_successes == all_args.num_agents)

        results["episode_rewards"].append(episode_reward)
        results["episode_lengths"].append(step)
        results["successes"].append(1 if success else 0)
        results["collisions"].append(1 if collision else 0)
        results["timeouts"].append(1 if timeout else 0)
        results["agent_success_count"].append(agent_successes)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}, Avg Reward: {np.mean(results['episode_rewards']):.2f}")

    # 成功率计算方式：成功到达目标的智能体总数 / (episode数 * 智能体数)
    total_agent_successes = sum(results["agent_success_count"])
    total_agent_attempts = num_episodes * all_args.num_agents
    results["success_rate"] = total_agent_successes / total_agent_attempts
    results["collision_rate"] = np.mean(results["collisions"])
    results["timeout_rate"] = np.mean(results["timeouts"])

    return results


def parser_args_eval(args, parser):
    # 使用不同的参数名避免与config.py中已有参数冲突
    parser.add_argument("--log_dir", type=str, default="/home/inno/proj/MAPPO/results/train/run24/logs")
    parser.add_argument("--save_dir", type=str, default="/home/inno/proj/MAPPO/results/eval/run24")
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    eval_args = parser_args_eval(args, parser)

    # 合并参数
    MODEL_DIR = all_args.model_dir if all_args.model_dir else "/home/inno/proj/MAPPO/results/train/run24/models"
    LOG_DIR = eval_args.log_dir
    SAVE_DIR = eval_args.save_dir
    NUM_EVAL_EPISODES = eval_args.num_eval_episodes

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("MAPPO Model Evaluation")
    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Save to: {SAVE_DIR}")
    print("=" * 60)

    # 加载并绘制训练曲线
    logs = load_training_logs(LOG_DIR)
    if logs:
        plot_training_curves(logs, SAVE_DIR)

    # 配置参数
    parser = get_config()
    all_args = parser_args(args, parser)

    all_args.num_agents = 15
    all_args.num_humans = 2
    all_args.num_attention_agents = 10
    all_args.n_rollout_threads = 1
    all_args.episode_length = 400
    all_args.model_dir = MODEL_DIR
    all_args.method = "ppo"
    all_args.use_render = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建环境
    print("Creating environment...")
    envs = make_eval_env(all_args)

    # 加载模型
    print("Loading model...")
    from policy.mappo.MAPPOPolicy import RMAPPOPolicy

    share_obs_space = envs.share_observation_space[0]

    policy = RMAPPOPolicy(
        all_args,
        envs.robot_observation_space[0],
        envs.human_observation_space[0],
        share_obs_space,
        envs.action_space[0],
        device=device
    )

    policy.actor.load_state_dict(torch.load(os.path.join(MODEL_DIR, "actor.pt"), map_location=device))
    policy.critic.load_state_dict(torch.load(os.path.join(MODEL_DIR, "critic.pt"), map_location=device))
    policy.actor.eval()
    policy.critic.eval()
    print("Model loaded successfully")

    # 评估模型
    results = evaluate_model(all_args, envs, policy, NUM_EVAL_EPISODES)

    # 保存结果
    with open(os.path.join(SAVE_DIR, "evaluation_results.json"), "w") as f:
        json.dump({k: v for k, v in results.items() if isinstance(v, (int, float, list))}, f, indent=2)

    # 绘制评估结果
    plot_evaluation_results(results, SAVE_DIR)

    # 打印摘要
    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Avg Reward: {np.mean(results['episode_rewards']):.4f} +/- {np.std(results['episode_rewards']):.4f}")
    print(f"Agent Success Rate: {results['success_rate']*100:.2f}% ({sum(results['agent_success_count'])}/{NUM_EVAL_EPISODES * all_args.num_agents} agents)")
    print(f"Episode Success Rate: {np.mean(results['successes'])*100:.2f}% (all agents reached goal)")
    print(f"Collision Rate: {results['collision_rate']*100:.2f}%")
    print(f"Timeout Rate: {results['timeout_rate']*100:.2f}%")
    print(f"Avg Episode Length: {np.mean(results['episode_lengths']):.1f} steps")
    print("=" * 60)

    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])