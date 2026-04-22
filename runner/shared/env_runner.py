"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from runner.shared.base_runner import Runner
from envs.utils.utils import reach_goal
import imageio

# Summed per env step into episode_meta (dynamic / undetermined / static); excludes shared_team to avoid n-fold count.
_EPISODE_META_SHAPING_KEYS = (
    "c_nav",
    "c_prox",
    "c_goal",
    "c_formation",
    "c_avoid",
    "c_explore_undervisible",
    "c_cluster_shaping",
    "c_ctde_remaining_target",
    "c_claimed_target",
    "c_reciprocal_swap",
    "c_goal_contention",
    "c_low_density_explore",
    "undetermined_hungarian_bonus",
)


def _episode_meta_shaping_from_infos(infos) -> float:
    total = 0.0
    if infos is None:
        return total
    for inf in np.asarray(infos, dtype=object).ravel():
        if inf is None:
            continue
        rt = getattr(inf, "reward_terms", None)
        if not isinstance(rt, dict):
            continue
        for k in _EPISODE_META_SHAPING_KEYS:
            v = rt.get(k, 0.0)
            if isinstance(v, (int, float, np.floating)):
                total += float(v)
    return total


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
    "formation_time_w",
    "c_avoid",
    "c_nav",
    "c_goal",
    "reward_shaped",
    "r_attn_comm_raw",
    "c_attn_comm",
    "attn_comm_align",
    "attn_comm_diversity",
    "attn_comm_smooth_term",
    "nd_terminal_tr",
    "nd_tr_applied",
    "goal_entered_from_outside",
    "goal_flag",
    "shared_team",
    "dynamic_team_dist_now",
    "dynamic_team_dist_delta",
    "r_prox_raw",
    "r_arrive_raw",
    "c_prox",
    "dynamic_conflict_weighted",
    "dynamic_crowding_sq",
    "c_dynamic_crowding",
    "dynamic_switch_near_goal_extra",
    "dynamic_nav_scale",
    "r_explore_undervisible_raw",
    "dynamic_explore_shortfall",
    "dynamic_explore_n_in_view",
    "dynamic_explore_align",
    "dynamic_explore_density_mult",
    "dynamic_local_neighbors",
    "dynamic_sparse_urgency_mult",
    "dynamic_same_target_contested",
    "dynamic_chasing_claimed",
    "r_claimed_target_raw",
    "c_claimed_target",
    "r_low_density_explore_raw",
    "dynamic_low_density_mode",
    "c_low_density_explore",
    "c_explore_undervisible",
    "dynamic_in_reciprocal_swap",
    "r_reciprocal_swap_raw",
    "c_reciprocal_swap",
    "dynamic_goal_contention_excess",
    "c_goal_contention",
    "dist_nearest_unclaimed",
    "pre_dist_nearest_unclaimed",
    "r_ctde_remaining_raw",
    "c_ctde_remaining_target",
    "dynamic_reward_v2",
    "dynamic_cluster_role",
    "r_cluster_shaping_raw",
    "r_cluster_dispersion_raw",
    "c_cluster_shaping",
    "undetermined_dist_penalty_raw",
    "undetermined_dist_penalty_quad_raw",
    "nd_progress_delta_applied",
    "nd_timeout_no_goal_penalty_raw",
    "undetermined_hungarian_bonus",
    "laplacian_S_L",
    "undetermined_v2_sl_raw",
    "undetermined_v2_sl_delta_raw",
    "undetermined_v2_sl_success",
    "undetermined_v2_sl_type2_crossing",
    "undetermined_v2_sl_literal_relax_w",
    "undetermined_v2_sl_pre_success_step_raw",
    "undetermined_v2_sl_pre_success_travel_raw",
    "nd_goal_leave_penalty_raw",
    "reward_final",
)


def _render_numeric_reward_fields(terms: dict) -> dict[str, float]:
    """Scalar numeric fields from info.reward_terms (for episode sums / means)."""
    out: dict[str, float] = {}
    if not isinstance(terms, dict):
        return out
    for k, v in terms.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _render_init_reward_accum(n_envs: int, n_agents: int):
    return [[defaultdict(float) for _ in range(n_agents)] for _ in range(n_envs)]


def _render_init_last_reward_modes(n_envs: int, n_agents: int):
    return [[None for _ in range(n_agents)] for _ in range(n_envs)]


def _normalize_infos_per_env(infos) -> list:
    """VecEnv may return tuple(list) or legacy ndarray (1, n_agents) object; -> list[list[info]]."""
    if infos is None:
        return []
    if isinstance(infos, tuple) and len(infos) > 0:
        first = infos[0]
        if isinstance(first, (list, tuple)):
            return [list(x) if isinstance(x, (list, tuple)) else list(x) for x in infos]
        if isinstance(first, np.ndarray):
            return [list(np.asarray(x, dtype=object).ravel()) for x in infos]
    arr = np.asarray(infos, dtype=object)
    if arr.ndim == 2:
        return [list(arr[i].ravel()) for i in range(arr.shape[0])]
    if arr.ndim == 1:
        if arr.size == 0:
            return []
        el0 = arr.flat[0]
        if isinstance(el0, (list, tuple, np.ndarray)):
            return [list(np.asarray(x, dtype=object).ravel()) for x in arr]
        return [list(arr)]
    return []


def _accumulate_render_reward_step(accum, last_modes, infos, rewards_arr: np.ndarray) -> None:
    if infos is None:
        return
    env_infos = _normalize_infos_per_env(infos)
    rew = np.asarray(rewards_arr)
    n_e = min(len(env_infos), int(rew.shape[0]), len(accum))
    for env_i in range(n_e):
        per_agent = env_infos[env_i]
        if not isinstance(per_agent, (list, tuple)):
            continue
        row = accum[env_i]
        lm = last_modes[env_i]
        for aid, inf in enumerate(per_agent):
            if aid >= len(row):
                break
            try:
                row[aid]["reward_env"] += float(rew[env_i, aid, 0])
            except (IndexError, TypeError, ValueError):
                pass
            terms = getattr(inf, "reward_terms", None)
            if not isinstance(terms, dict):
                continue
            rm = terms.get("reward_mode")
            if isinstance(rm, str):
                lm[aid] = rm
            for k, fv in _render_numeric_reward_fields(terms).items():
                row[aid][k] += fv


def _build_render_episode_reward_stats(
    episode_1based: int,
    episode_length: int,
    pattern_name: str,
    accum,
    last_modes,
) -> dict:
    T = max(1, int(episode_length))
    sample_mode = None
    envs_out = []
    for env_i, agents in enumerate(accum):
        block = {"env_id": int(env_i), "agents": []}
        for aid, sdict in enumerate(agents):
            sm = {k: float(v) for k, v in dict(sdict).items()}
            mean = {k: sm[k] / T for k in sm}
            rm = last_modes[env_i][aid] if env_i < len(last_modes) and aid < len(last_modes[env_i]) else None
            if sample_mode is None and isinstance(rm, str):
                sample_mode = rm
            block["agents"].append(
                {
                    "agent_id": int(aid),
                    "reward_mode": rm,
                    "sum": sm,
                    "mean": mean,
                }
            )
        envs_out.append(block)
    return {
        "episode": int(episode_1based),
        "episode_length": T,
        "pattern": str(pattern_name),
        "reward_mode_sample": sample_mode,
        "envs": envs_out,
    }


def _t2n(x):
    return x.detach().cpu().numpy()


def _comm_broadcasts_to_env_shape(broadcast_msg, n_rollout_threads, num_agents):
    """
    Actor returns (n_envs * n_agents, msg_dim). VecEnv / EnvCore expect
    (n_envs, n_agents, msg_dim) so SubprocVecEnv can send one slice per worker.
    """
    b = np.asarray(broadcast_msg, dtype=np.float32)
    if b.ndim == 3:
        return b
    if b.ndim == 2 and b.shape[0] == n_rollout_threads * num_agents:
        return b.reshape(n_rollout_threads, num_agents, b.shape[1])
    if b.ndim == 2 and n_rollout_threads == 1:
        return b[np.newaxis, ...]
    return b


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
        env_infos = _normalize_infos_per_env(infos)
        if not env_infos:
            return
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

    def _undetermined_needs_resolve(self, obs, infos):
        if not getattr(self.all_args, "enable_undetermined_goal", False):
            return False
        rod = int(self.all_args.robot_obs_dim)
        pend_idx = rod - 1
        if np.any(obs[:, :, 0, pend_idx] > 0.5):
            return True
        if infos is None:
            return False
        env_infos = _normalize_infos_per_env(infos)
        for env_i in range(min(len(env_infos), self.n_rollout_threads)):
            per_agent = env_infos[env_i]
            if not isinstance(per_agent, (list, tuple)):
                continue
            for aid in range(min(len(per_agent), self.num_agents)):
                inf = per_agent[aid]
                if getattr(inf, "undetermined_need_target", False):
                    return True
        return False

    @torch.no_grad()
    def _undetermined_resolve(self, obs):
        if not hasattr(self.envs, "apply_undetermined_targets"):
            return obs
        max_r = int(getattr(self.all_args, "undetermined_max_auction_rounds", 8))
        n_envs = self.n_rollout_threads
        n_agents = self.num_agents
        robot_obs_dim = self.all_args.robot_obs_dim + 2
        out = obs
        self.trainer.prep_rollout()
        for _ in range(max_r):
            robot_rows = out[:, :, 0, :robot_obs_dim]
            flat = robot_rows.reshape(n_envs * n_agents, robot_obs_dim)
            tid = self.trainer.policy.get_undetermined_targets(flat, deterministic=False)
            tid = _t2n(tid).astype(np.int64).reshape(n_envs, n_agents)
            out = self.envs.apply_undetermined_targets(tid)
            pend = out[:, :, 0, int(self.all_args.robot_obs_dim) - 1]
            if not np.any(pend > 0.5):
                break
        return out

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
                        broadcast_msg,
                        comm_rnn_out,
                        undet_logits,
                    ) = self.collect(step)

                    if (
                        getattr(self.all_args, "use_attn_comm_actor", False)
                        and broadcast_msg is not None
                        and hasattr(self.envs, "set_comm_broadcasts")
                    ):
                        b = _comm_broadcasts_to_env_shape(
                            broadcast_msg, self.n_rollout_threads, self.num_agents
                        )
                        self.envs.set_comm_broadcasts(b)

                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions_env)
                    if getattr(self.all_args, "enable_undetermined_goal", False) and self._undetermined_needs_resolve(
                        obs, infos
                    ):
                        obs = self._undetermined_resolve(obs)
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
                        comm_rnn_out,
                        undet_logits,
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
        if getattr(self.all_args, "enable_undetermined_goal", False):
            obs = self._undetermined_resolve(obs)

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
        if self.buffer.comm_rnn_states is not None:
            self.buffer.comm_rnn_states[0].fill(0.0)

    @torch.no_grad()
    def collect(self, step):
        # print(f'========={self.buffer.obs.shape}=========')
        # print(f'================={np.concatenate(self.buffer.obs[step,:,:1,:,:]).shape},{np.concatenate(self.buffer.obs[step,:,1:,:,:]).shape}=====================')
        # buffer.obs shape: (T+1, n_rollout_threads, num_agents, R, C)
        # Extract per-agent robot row (row 0) and human rows (1:1+human_num),
        # then reshape to batch form: batch = n_rollout_threads * num_agents
        obs_raw = self.buffer.obs[step]  # shape: (n_rollout_threads, num_agents, R, C)
        if getattr(self.all_args, "enable_undetermined_goal", False) and self._undetermined_needs_resolve(
            obs_raw, None
        ):
            obs_raw = self._undetermined_resolve(obs_raw)
            self.buffer.obs[step] = obs_raw
            if self.use_centralized_V:
                share_obs = obs_raw.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
                self.buffer.share_obs[step] = share_obs
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
        comm_flat = None
        if self.buffer.comm_rnn_states is not None:
            comm_flat = np.concatenate(self.buffer.comm_rnn_states[step])
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
            comm_rnn_out,
            broadcast_msg,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            robot_obs,
            human_obs,
            # np.concatenate(self.buffer.robot_obs[step]),
            # np.concatenate(self.buffer.human_obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            comm_rnn_states_actor=comm_flat,
        )
        undet_logits_out = None
        if getattr(self.buffer, "undet_target_logits", None) is not None:
            lt = self.trainer.policy.get_undetermined_target_logits(
                torch.from_numpy(robot_obs).float().to(self.device)
            )
            undet_logits_out = np.array(
                np.split(_t2n(lt), self.n_rollout_threads)
            ).astype(np.float32)
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
        bm = _t2n(broadcast_msg) if broadcast_msg is not None else None
        cr = None
        if self.buffer.comm_rnn_states is not None:
            cr = np.array(np.split(_t2n(comm_rnn_out), self.n_rollout_threads))
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
            bm,
            cr,
            undet_logits_out,
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
            comm_rnn_out,
            undet_logits,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        if self.buffer.comm_rnn_states is not None and comm_rnn_out is not None:
            comm_rnn_out = np.asarray(comm_rnn_out, dtype=np.float32)
            comm_rnn_out[dones == True] = np.zeros(
                ((dones == True).sum(), self.buffer.comm_state_dim),
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
            comm_rnn_states_actor=comm_rnn_out if self.buffer.comm_rnn_states is not None else None,
            undet_target_logits=undet_logits,
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
        reward_dir = Path(self.run_dir) / "reward"
        reward_dir.mkdir(parents=True, exist_ok=True)
        render_reward_stats_path = reward_dir / "render_reward_stats.jsonl"
        with open(render_reward_stats_path, "w", encoding="utf-8") as _rs:
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
                gp = getattr(core_env, "goal_positions", None)
                if gp:
                    episode_meta_record["goal_positions"] = [
                        [float(x), float(y)] for (x, y) in core_env.goal_positions
                    ]
                if getattr(core_env, "dynamic_goal_assignment", False) and gp:
                    episode_meta_record["dynamic_target"] = True
                if getattr(core_env, "undetermined_goal_assignment", False) and gp:
                    episode_meta_record["undetermined_goal"] = True
                    if getattr(core_env, "undetermined_goal_v3", False):
                        episode_meta_record["undetermined_goal_v3"] = True
                    elif getattr(core_env, "undetermined_goal_v2", False):
                        episode_meta_record["undetermined_goal_v2"] = True
                if not episode_meta_record.get("dynamic_target") and not episode_meta_record.get(
                    "undetermined_goal"
                ):
                    episode_meta_record["static_target_assignment"] = True
                    episode_meta_record["goal_positions"] = [
                        [float(r.gx), float(r.gy)] for r in core_env.robots
                    ]
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
            rnn_states_critic = np.zeros_like(rnn_states)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            comm_rnn = None
            if getattr(self.all_args, "use_attn_comm_actor", False):
                comm_rnn = np.zeros(
                    (
                        self.n_rollout_threads,
                        self.num_agents,
                        int(self.trainer.policy.actor.comm_state_dim),
                    ),
                    dtype=np.float32,
                )

            episode_rewards = []
            # buffer to store (px, py) per step: shape = (episode_length, n_envs, num_agents, 2)
            episode_coords = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
            episode_succes = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 1))
            target_ids_trace = []
            episode_shaping_ref = 0.0
            reward_accum = _render_init_reward_accum(self.n_rollout_threads, self.num_agents)
            reward_last_modes = _render_init_last_reward_modes(self.n_rollout_threads, self.num_agents)

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                try:
                    _ce = self.envs.env.env
                    _gp = getattr(_ce, "goal_positions", None)
                    _dyn = getattr(_ce, "dynamic_goal_assignment", False)
                    _und = getattr(_ce, "undetermined_goal_assignment", False)
                    _nr = len(_ce.robots)
                    if _dyn or _und:
                        if _gp:
                            _K = max(1, int(getattr(_ce, "num_goal_targets", 1)))
                            target_ids_trace.append(
                                [int(r.target_id) % _K for r in _ce.robots]
                            )
                    else:
                        _K = max(1, _nr)
                        target_ids_trace.append([i % _K for i in range(_nr)])
                except Exception:
                    pass
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

                if self.use_centralized_V:
                    share_obs = obs.reshape(self.n_rollout_threads, -1)
                    share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
                else:
                    share_obs = obs
                cent_flat = np.concatenate(share_obs)
                comm_flat = np.concatenate(comm_rnn) if comm_rnn is not None else None
                (
                    _v,
                    action,
                    _lp,
                    rnn_states,
                    rnn_states_critic,
                    comm_rnn_next,
                    bm,
                ) = self.trainer.policy.get_actions(
                    cent_flat,
                    render_robot_obs,
                    render_human_obs,
                    np.concatenate(rnn_states),
                    np.concatenate(rnn_states_critic),
                    np.concatenate(masks),
                    deterministic=True,
                    comm_rnn_states_actor=comm_flat,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
                if comm_rnn is not None:
                    comm_rnn = np.array(np.split(_t2n(comm_rnn_next), self.n_rollout_threads))
                if (
                    bm is not None
                    and getattr(self.all_args, "use_attn_comm_actor", False)
                    and hasattr(self.envs, "set_comm_broadcasts")
                ):
                    b = _comm_broadcasts_to_env_shape(
                        _t2n(bm), self.n_rollout_threads, self.num_agents
                    )
                    self.envs.set_comm_broadcasts(b)

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
                _accumulate_render_reward_step(reward_accum, reward_last_modes, infos, rewards)
                if episode_meta_record is not None and (
                    episode_meta_record.get("dynamic_target")
                    or episode_meta_record.get("undetermined_goal")
                    or episode_meta_record.get("static_target_assignment")
                ):
                    episode_shaping_ref += _episode_meta_shaping_from_infos(infos)
                # Also mark success after physics: agents can enter the goal during step().
                for i, robot in enumerate(self.envs.env.env.robots):
                    if reach_goal(robot):
                        episode_succes[step, 0, i, 0] = 1

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                rnn_states_critic[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                if comm_rnn is not None:
                    comm_rnn[dones == True] = np.zeros(
                        ((dones == True).sum(), int(self.trainer.policy.actor.comm_state_dim)),
                        dtype=np.float32,
                    )

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
                _dt = episode_meta_record.get("dynamic_target")
                _ug = episode_meta_record.get("undetermined_goal")
                _st = episode_meta_record.get("static_target_assignment")
                if len(target_ids_trace) == int(self.episode_length):
                    episode_meta_record["target_ids_by_step"] = target_ids_trace
                if _dt or _ug or _st:
                    episode_meta_record["shaping_reward_reference"] = float(episode_shaping_ref)
                with open(meta_path, "a", encoding="utf-8") as mf:
                    mf.write(
                        json.dumps(episode_meta_record, ensure_ascii=False) + "\n"
                    )

            ep_pattern = pattern_name
            if episode_meta_record is not None:
                ep_pattern = str(episode_meta_record.get("pattern", pattern_name))
            stats_rec = _build_render_episode_reward_stats(
                episode + 1,
                int(self.episode_length),
                ep_pattern,
                reward_accum,
                reward_last_modes,
            )
            with open(render_reward_stats_path, "a", encoding="utf-8") as rsf:
                rsf.write(json.dumps(stats_rec, ensure_ascii=False) + "\n")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        try:
            mp = Path(meta_path)
            if mp.is_file() and mp.stat().st_size > 0:
                rows_out = []
                with mp.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows_out.append(json.loads(line))
                with mp.with_name("episode_meta.json").open("w", encoding="utf-8") as jf:
                    json.dump(rows_out, jf, ensure_ascii=False, indent=2)
        except Exception:
            pass

        try:
            rsp = Path(self.run_dir) / "reward" / "render_reward_stats.jsonl"
            if rsp.is_file() and rsp.stat().st_size > 0:
                stats_rows = []
                with rsp.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            stats_rows.append(json.loads(line))
                with rsp.with_name("render_reward_stats.json").open("w", encoding="utf-8") as jf:
                    json.dump(stats_rows, jf, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
