"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from runner.shared.base_runner import Runner
from envs.utils.utils import reach_goal
import imageio


# Columns for reward_terms.csv (static + dynamic keys; unused cells left blank).
REWARD_TERMS_CSV_COLUMNS = (
    "episode",
    "step",
    "env_id",
    "agent_id",
    "reward_env",
    "info_type",
    "reward_mode",
    "r_avoid_raw",
    "r_formation_raw",
    "r_nav_raw",
    "r_goal_raw",
    "r_bonus_raw",
    "c_formation",
    "c_avoid",
    "c_nav",
    "c_goal",
    "reward_shaped",
    "nd_terminal_tr",
    "nd_tr_applied",
    "goal_entered_from_outside",
    "goal_flag",
    "shared_team",
    "reward_final",
)


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self._reward_terms_fp = None
        self._reward_terms_writer = None

    def _ensure_reward_terms_csv(self):
        if self._reward_terms_fp is not None:
            return
        # Same directory as TensorBoard (SummaryWriter); created in base_runner.__init__
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "reward_terms.csv"
        self._reward_terms_fp = open(path, "w", newline="", encoding="utf-8")
        self._reward_terms_writer = csv.DictWriter(
            self._reward_terms_fp,
            fieldnames=REWARD_TERMS_CSV_COLUMNS,
            extrasaction="ignore",
            restval="",
        )
        self._reward_terms_writer.writeheader()
        self._reward_terms_fp.flush()

    def _close_reward_terms_log(self):
        if self._reward_terms_fp is not None:
            self._reward_terms_fp.close()
            self._reward_terms_fp = None
            self._reward_terms_writer = None

    def _maybe_log_reward_terms(self, episode, step, infos, rewards):
        if not getattr(self.all_args, "save_reward_terms", False):
            return
        # Align with TensorBoard / console logging (same episodes as log_train)
        _li = max(1, int(self.log_interval))
        if int(episode) % _li != 0:
            return
        stride = max(1, int(getattr(self.all_args, "reward_terms_log_stride", 1)))
        if step % stride != 0:
            return
        max_envs = max(0, int(getattr(self.all_args, "reward_terms_max_envs", 1)))
        if max_envs == 0:
            return
        self._ensure_reward_terms_csv()
        if isinstance(infos, tuple):
            env_infos = infos
        else:
            env_infos = tuple(infos)
        n_e = min(max_envs, len(env_infos), int(rewards.shape[0]))
        for env_i in range(n_e):
            per_agent = env_infos[env_i]
            if not isinstance(per_agent, (list, tuple)):
                continue
            for aid, inf in enumerate(per_agent):
                terms = getattr(inf, "reward_terms", None) or {}
                row = {
                    "episode": episode,
                    "step": step,
                    "env_id": env_i,
                    "agent_id": aid,
                    "reward_env": float(rewards[env_i, aid, 0]),
                    "info_type": type(inf).__name__,
                }
                for k in REWARD_TERMS_CSV_COLUMNS:
                    if k in terms:
                        row[k] = terms[k]
                self._reward_terms_writer.writerow(row)
        self._reward_terms_fp.flush()

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        try:
            for episode in range(episodes):
                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)

                for step in range(self.episode_length):
                    # Sample actions
                    (
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                        actions_env,
                    ) = self.collect(step)

                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions_env)
                    self._maybe_log_reward_terms(episode, step, infos, rewards)

                    data = (
                        obs,
                        rewards,
                        dones,
                        infos,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )

                    # insert data into buffer
                    self.insert(data)

                # compute return and update network
                self.compute()
                train_infos = self.train()

                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                # save model
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save()

                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print(
                        "\n Algo MAPPO Exp test updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                            # self.algorithm_name,
                            # self.experiment_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start)),
                        )
                    )
                    # print(
                    #     "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                    #         self.all_args.scenario_name,
                    #         self.algorithm_name,
                    #         self.experiment_name,
                    #         episode,
                    #         episodes,
                    #         total_num_steps,
                    #         self.num_env_steps,
                    #         int(total_num_steps / (end - start)),
                    #     )
                    # )

                    # if self.env_name == "MPE":
                    #     env_infos = {}
                    #     for agent_id in range(self.num_agents):
                    #         idv_rews = []
                    #         for info in infos:
                    #             if 'individual_reward' in info[agent_id].keys():
                    #                 idv_rews.append(info[agent_id]['individual_reward'])
                    #         agent_k = 'agent%i/individual_rewards' % agent_id
                    #         env_infos[agent_k] = idv_rews

                    train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                    self.log_train(train_infos, total_num_steps)
                    # self.log_env(env_infos, total_num_steps)

                # eval
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)
        finally:
            self._close_reward_terms_log()

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = shape = [env_num, agent_num， agent_num * obs_dim]
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        # print(f'========={self.buffer.obs.shape}=========')
        # print(f'================={np.concatenate(self.buffer.obs[step,:,:1,:,:]).shape},{np.concatenate(self.buffer.obs[step,:,1:,:,:]).shape}=====================')
        # buffer.obs shape: (T+1, n_rollout_threads, num_agents, R, C)
        # Extract per-agent robot row (row 0) and human rows (1:1+human_num),
        # then reshape to batch form: batch = n_rollout_threads * num_agents
        obs_raw = self.buffer.obs[step]  # shape: (n_rollout_threads, num_agents, R, C)
        n_envs = self.n_rollout_threads
        n_agents = self.num_agents
        robot_obs_dim = self.all_args.robot_obs_dim + 2
        human_obs_dim = self.all_args.human_obs_dim
        human_num = self.all_args.num_humans
        use_human_obs = getattr(self.all_args, "use_human_obs", True)

        # robot local obs is the 0-th row
        robot_rows = obs_raw[:, :, 0, :robot_obs_dim]  # (n_envs, n_agents, robot_obs_dim)
        robot_obs = robot_rows.reshape(n_envs * n_agents, robot_obs_dim)  # (batch, robot_obs_dim)

        # human obs are rows 1..human_num (optional)
        if use_human_obs and human_num > 0:
            human_rows = obs_raw[:, :, 1:1 + human_num, :human_obs_dim]  # (n_envs, n_agents, human_num, human_obs_dim)
            human_obs = human_rows.reshape(n_envs * n_agents, human_num, human_obs_dim)  # (batch, human_num, human_obs_dim)
        else:
            human_obs = np.zeros((n_envs * n_agents, 0, human_obs_dim), dtype=np.float32)
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            robot_obs,
            human_obs,
            # np.concatenate(self.buffer.robot_obs[step]),
            # np.concatenate(self.buffer.human_obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            # actions shape: [n_envs, num_agents, dims]
            actions_env = actions[0] if self.n_rollout_threads == 1 else actions
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # keep previous behavior but remove leading env dim for single-env case
            tmp = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
            actions_env = tmp[0] if self.n_rollout_threads == 1 else tmp
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    # @torch.no_grad()
    # def render(self):
    #     """Visualize the env."""
    #     envs = self.envs

    #     all_frames = []
    #     for episode in range(self.all_args.render_episodes):
    #         obs = envs.reset()
    #         if self.all_args.save_gifs:
    #             image = envs.render("rgb_array")[0][0]
    #             all_frames.append(image)
    #         else:
    #             envs.render("human")

    #         rnn_states = np.zeros(
    #             (
    #                 self.n_rollout_threads,
    #                 self.num_agents,
    #                 self.recurrent_N,
    #                 self.hidden_size,
    #             ),
    #             dtype=np.float32,
    #         )
    #         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

    #         episode_rewards = []

    #         for step in range(self.episode_length):
    #             calc_start = time.time()

    #             self.trainer.prep_rollout()
    #             # Split obs into robot-local and human observations (same convention as collect())
    #             obs_raw = obs  # shape: (n_rollout_threads, num_agents, R, C)
    #             n_envs = self.n_rollout_threads
    #             n_agents = self.num_agents
    #             robot_obs_dim = self.all_args.robot_obs_dim + 2
    #             human_obs_dim = self.all_args.human_obs_dim
    #             human_num = self.all_args.num_humans

    #             robot_rows = obs_raw[:, :, 0, :robot_obs_dim]  # (n_envs, n_agents, robot_obs_dim)
    #             render_robot_obs = robot_rows.reshape(n_envs * n_agents, robot_obs_dim)

    #             human_rows = obs_raw[:, :, 1:1 + human_num, :human_obs_dim]  # (n_envs, n_agents, human_num, human_obs_dim)
    #             render_human_obs = human_rows.reshape(n_envs * n_agents, human_num, human_obs_dim)

    #             action, rnn_states = self.trainer.policy.act(
    #                 render_robot_obs,
    #                 render_human_obs,
    #                 np.concatenate(rnn_states),
    #                 np.concatenate(masks),
    #                 deterministic=True,
    #             )
    #             actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
    #             rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

    #             if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
    #                 actions_env = actions[0] if self.n_rollout_threads == 1 else actions
    #             elif envs.action_space[0].__class__.__name__ == "Discrete":
    #                 tmp = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
    #                 actions_env = tmp[0] if self.n_rollout_threads == 1 else tmp
    #             else:
    #                 raise NotImplementedError
    #             # print(actions_env)
    #             # Obser reward and next obs
    #             obs, rewards, dones, infos = envs.step(actions_env)
    #             episode_rewards.append(rewards)

    #             rnn_states[dones == True] = np.zeros(
    #                 ((dones == True).sum(), self.recurrent_N, self.hidden_size),
    #                 dtype=np.float32,
    #             )
    #             masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #             masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

    #             if self.all_args.save_gifs:
    #                 image = envs.render("rgb_array")[0][0]
    #                 all_frames.append(image)
    #                 calc_end = time.time()
    #                 elapsed = calc_end - calc_start
    #                 if elapsed < self.all_args.ifi:
    #                     time.sleep(self.all_args.ifi - elapsed)
    #             else:
    #                 envs.render("human")

    #         print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

    #     if self.all_args.save_gifs:
    #         imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    @torch.no_grad()
    def render(self):
        """Visualize the env and save per-step robot coordinates to files."""
        import os

        envs = self.envs

        # prepare coords directory and files (one file per agent)
        coords_dir = str(self.run_dir / "coords")
        if not os.path.exists(coords_dir):
            os.makedirs(coords_dir)
        coords_files = [os.path.join(coords_dir, f"coords_agent{i}.txt") for i in range(self.num_agents)]

        success_dir = str(self.run_dir / "succ")
        if not os.path.exists(success_dir):
            os.makedirs(success_dir)
        success_files = [os.path.join(success_dir, f"success_agent{i}.txt") for i in range(self.num_agents)]

        # write render metadata (font pattern name + agent count) into output txt files
        pattern_name = "unknown"
        pattern_template_len = 0
        try:
            # DummyVecEnv: self.envs.env is DiscreteActionEnv, .env is EnvCore
            core_env = self.envs.env.env
            pattern_name = getattr(core_env, "pattern_name", pattern_name)
            st = getattr(core_env, "s_shape_targets", None) or []
            pattern_template_len = len(st)
        except Exception:
            pass
        meta_path = os.path.join(str(self.run_dir), "episode_meta.jsonl")
        with open(meta_path, "w", encoding="utf-8") as _mf:
            pass
        header_line2 = (
            f"# rollout_length={int(self.episode_length)}, "
            f"pattern_template_len={pattern_template_len}, "
            f"meta_file=episode_meta.jsonl\n"
        )
        for fpath in coords_files + success_files:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"# agents={self.num_agents}, pattern={pattern_name}\n")
                f.write(header_line2)
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            episode_meta_record = None
            try:
                core_env = self.envs.env.env
                episode_meta_record = {
                    "episode": int(episode + 1),
                    "pattern": str(getattr(core_env, "pattern_name", pattern_name)),
                    "episode_length": int(self.episode_length),
                    "pattern_template_len": int(
                        len(getattr(core_env, "s_shape_targets", None) or [])
                    ),
                    "agent_goals": [
                        [float(r.gx), float(r.gy)] for r in core_env.robots
                    ],
                }
            except Exception:
                pass
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            # buffer to store (px, py) per step: shape = (episode_length, n_envs, num_agents, 2)
            episode_coords = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
            episode_succes = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 1))

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                # Split obs into robot-local and human observations (same convention as collect())
                obs_raw = obs  # shape: (n_rollout_threads, num_agents, R, C)
                n_envs = self.n_rollout_threads
                n_agents = self.num_agents
                robot_obs_dim = self.all_args.robot_obs_dim + 2
                human_obs_dim = self.all_args.human_obs_dim
                human_num = self.all_args.num_humans
                use_human_obs = getattr(self.all_args, "use_human_obs", True)

                robot_rows = obs_raw[:, :, 0, :robot_obs_dim]  # (n_envs, n_agents, robot_obs_dim)
                # last two entries in robot_rows are px, py
                episode_coords[step] = robot_rows[:, :, -2:]
                for i, robot in enumerate(self.envs.env.env.robots):
                    if reach_goal(robot):
                        episode_succes[step, 0, i, 0] = 1

                render_robot_obs = robot_rows.reshape(n_envs * n_agents, robot_obs_dim)

                if use_human_obs and human_num > 0:
                    human_rows = obs_raw[:, :, 1:1 + human_num, :human_obs_dim]  # (n_envs, n_agents, human_num, human_obs_dim)
                    render_human_obs = human_rows.reshape(n_envs * n_agents, human_num, human_obs_dim)
                else:
                    render_human_obs = np.zeros((n_envs * n_agents, 0, human_obs_dim), dtype=np.float32)

                action, rnn_states = self.trainer.policy.act(
                    render_robot_obs,
                    render_human_obs,
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    actions_env = actions[0] if self.n_rollout_threads == 1 else actions
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    tmp = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                    actions_env = tmp[0] if self.n_rollout_threads == 1 else tmp
                else:
                    raise NotImplementedError
                # print(actions_env)
                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)
                # Also mark success after physics: agents can enter the goal during step().
                for i, robot in enumerate(self.envs.env.env.robots):
                    if reach_goal(robot):
                        episode_succes[step, 0, i, 0] = 1

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            # write coordinates for this episode: one line per agent, appended to each agent's file
            for agent_id in range(self.num_agents):
                with open(coords_files[agent_id], "a") as f:
                    env_chunks = []
                    for env_i in range(self.n_rollout_threads):
                        coords_list = ["{:.4f} {:.4f}".format(x, y) for x, y in episode_coords[:, env_i, agent_id, :]]
                        env_chunks.append("; ".join(coords_list))
                    # if multiple envs, separate them with ' | '
                    line = f"{episode+1}, " + " | ".join(env_chunks) + "\n"
                    f.write(line)
                with open(success_files[agent_id], "a") as f:
                    env_chunks = []
                    for env_i in range(self.n_rollout_threads):
                        coords_list = [("1" if episode_succes[x, env_i, agent_id, 0] else "0") for x in range(episode_succes.shape[0])]
                        env_chunks.append("; ".join(coords_list))  
                    line = f"{episode+1}, " + " | ".join(env_chunks) + "\n"
                    f.write(line)

            if episode_meta_record is not None:
                with open(meta_path, "a", encoding="utf-8") as mf:
                    mf.write(
                        json.dumps(episode_meta_record, ensure_ascii=False) + "\n"
                    )

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
