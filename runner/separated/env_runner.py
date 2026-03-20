import time
import os
import numpy as np
from itertools import chain
import torch
from pathlib import Path
from runner.separated.base_runner import Runner
from envs.utils.info import ReachGoal


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self, args):
        reward_dir = str(self.run_dir / "reward")
        if not os.path.exists(reward_dir):
            os.makedirs(reward_dir)
        reward_files = []
        for i in range(self.num_robots):
            reward_files.append(os.path.join(reward_dir, f"reward{i}.txt"))

        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_robots):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            episode_rewards = np.zeros([self.n_rollout_threads,self.num_robots,1])

            for step in range(self.episode_length):
                # print('step',step)
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
                obs, rewards, dones, infos = self.envs.step(actions)
                # obs = obs[:,:,:,:self.obs_dim]
                # print(rewards)
                # time.sleep(0.1)
                episode_rewards += rewards

                data = (obs,rewards,dones,infos,values,actions,action_log_probs,rnn_states,rnn_states_critic,)

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            # time1 = time.time()
            self.compute()
            train_infos = self.train()
            # print(time.time() - time1)
            # print('network has update!')
            for i in range(self.num_robots):
                with open(reward_files[i], "a") as f:
                    rewards_str = ', '.join(map(str, episode_rewards[:,i,:]))
                    line = f"{episode+1}, {rewards_str.replace('[', '').replace(']', '')}\n"
                    f.write(line)

            # compute success rate for this episode across parallel envs
            # infos is a sequence (len = n_rollout_threads) where each element is a list of per-agent info objects
            # success_count = 0
            # for env_info in infos:
            #     # env_info is a list (or iterable) of info objects for each agent in this environment
            #     all_agents_success = True
            #     for agent_id in range(self.num_robots):
            #         try:
            #             info_obj = env_info[agent_id]
            #         except Exception:
            #             # if structure is unexpected, skip counting this env
            #             all_agents_success = False
            #             break
            #         if not isinstance(info_obj, ReachGoal):
            #             all_agents_success = False
            #             break
            #     if all_agents_success:
            #         success_count += 1

            # success_rate = float(success_count) / float(self.n_rollout_threads)
            # # Log env-level metric so it will appear in tensorboard/summary.json
            # self.log_env({"success_rate": [success_rate]})
            ###############################################################
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MPE":
                    for agent_id in range(self.num_robots):
                        idv_rews = []
                        for info in infos:
                            if "individual_reward" in info[agent_id].keys():
                                idv_rews.append(info[agent_id]["individual_reward"])
                        train_infos[agent_id].update({"individual_rewards": np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length
                            }
                        )
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, robot_num, 1+human_num, obs_dim]
        # obs = obs[:,:,:,:self.obs_dim]
        share_obs = []
        share_obs1 = obs[:,:,0,:]
        share_obs2 = obs[:,0,1:,:]
        if self.num_humans > self.att_agents:
            share_obs2 = share_obs2[:,::-1,:]
        share_obs = np.concatenate([share_obs1, share_obs2], axis=1)

        for agent_id in range(self.num_robots):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].robot_obs[0] = np.array(list(obs[:,agent_id,0])).copy()
            self.buffer[agent_id].human_obs[0] = np.array(list(obs[:,agent_id,1:,:self.human_obs_dim])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_robots):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].robot_obs[step],
                self.buffer[agent_id].human_obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                action_env = action

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

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

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),dtype=np.float32)   
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_robots, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        share_obs1 = obs[:,:,0,:]
        share_obs2 = obs[:,0,1:,:]
        if self.num_humans > self.att_agents:
            share_obs2 = share_obs2[:,::-1,:]
        share_obs = np.concatenate([share_obs1, share_obs2], axis=1)

        for agent_id in range(self.num_robots):
            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id, 0])),
                np.array(list(obs[:, agent_id, 1:, :self.human_obs_dim])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],)
            

    # @torch.no_grad()
    # def eval(self, total_num_steps):
    #     eval_episode_rewards = []
    #     eval_obs = self.eval_envs.reset()

    #     eval_rnn_states = np.zeros(
    #         (
    #             self.n_eval_rollout_threads,
    #             self.num_robots,
    #             self.recurrent_N,
    #             self.hidden_size,
    #         ),
    #         dtype=np.float32,
    #     )
    #     eval_masks = np.ones((self.n_eval_rollout_threads, self.num_robots, 1), dtype=np.float32)

    #     for eval_step in range(self.episode_length):
    #         eval_temp_actions_env = []
    #         for agent_id in range(self.num_robots):
    #             self.trainer[agent_id].prep_rollout()
    #             eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
    #                 np.array(list(eval_obs[:, agent_id])),
    #                 eval_rnn_states[:, agent_id],
    #                 eval_masks[:, agent_id],
    #                 deterministic=True,
    #             )

    #             eval_action = eval_action.detach().cpu().numpy()
    #             # rearrange action
    #             if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
    #                 for i in range(self.eval_envs.action_space[agent_id].shape):
    #                     eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
    #                         eval_action[:, i]
    #                     ]
    #                     if i == 0:
    #                         eval_action_env = eval_uc_action_env
    #                     else:
    #                         eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
    #             elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
    #                 eval_action_env = np.squeeze(
    #                     np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
    #                 )
    #             else:
    #                 raise NotImplementedError

    #             eval_temp_actions_env.append(eval_action_env)
    #             eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

    #         # [envs, agents, dim]
    #         eval_actions_env = []
    #         for i in range(self.n_eval_rollout_threads):
    #             eval_one_hot_action_env = []
    #             for eval_temp_action_env in eval_temp_actions_env:
    #                 eval_one_hot_action_env.append(eval_temp_action_env[i])
    #             eval_actions_env.append(eval_one_hot_action_env)

    #         # Obser reward and next obs
    #         eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
    #         eval_episode_rewards.append(eval_rewards)

    #         eval_rnn_states[eval_dones == True] = np.zeros(
    #             ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
    #             dtype=np.float32,
    #         )
    #         eval_masks = np.ones((self.n_eval_rollout_threads, self.num_robots, 1), dtype=np.float32)
    #         eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

    #     eval_episode_rewards = np.array(eval_episode_rewards)

    #     eval_train_infos = []
    #     for agent_id in range(self.num_robots):
    #         eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
    #         eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
    #         print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

    #     self.log_train(eval_train_infos, total_num_steps)


    @torch.no_grad()
    def render(self, mode = 'vedio', visualize = False, method='ppo'):
        success_times = 0
        
        for episode in range(self.all_args.render_episodes):
            self.data_dir = str(self.run_dir / f"episode{episode+1}")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            agent_file = []
            for i in range(self.num_robots + self.num_humans):
                if i < self.num_robots:
                    agent_file.append(os.path.join(self.data_dir, f"robot{i+1}.txt"))
                else:
                    agent_file.append(os.path.join(self.data_dir, f"human{i-self.num_robots+1}.txt"))

            print('episode:',episode + 1)
            print('result saved in:',self.data_dir)
            episode_rewards = []
            obs = self.envs.reset()

            rnn_states = np.zeros((self.n_rollout_threads,self.num_robots,self.recurrent_N,self.hidden_size,),dtype=np.float32,)
            masks = np.ones((self.n_rollout_threads, self.num_robots, 1), dtype=np.float32)

            for step in range(self.episode_length):
                # print("step",step)
                actions = []
                if method == 'ppo':
                    # write robot obs
                    for i,o in enumerate(np.squeeze(obs)):
                        with open(agent_file[i], "a") as f:
                            line = f"{step+1}, {', '.join(map(str, o[0]))}\n"
                            f.write(line)
                    # write human obs
                    for i,o in enumerate(np.squeeze(obs)[0,:self.num_humans+1]): # abandon the zero array
                        if i == 0:
                            continue
                        with open(agent_file[i+self.num_robots-1], "a") as f:
                            line = f"{step+1}, {', '.join(map(str, o))}\n"
                            f.write(line)

                    # obs = obs[:,:,:,:self.obs_dim]
                    for agent_id in range(self.num_robots):
                        self.trainer[agent_id].prep_rollout()
                        #获取action
                        action, rnn_state = self.trainer[agent_id].policy.act(
                            np.array(list(obs[:, agent_id, 0])),      # robot_obs
                            np.array(list(obs[:, agent_id, 1:, :self.human_obs_dim])),       # humans_obs
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            deterministic=True,)
                        action = action.detach().cpu().numpy()
                        action = action[0]                        
                        actions.append(action)
                        rnn_states[:, agent_id] = _t2n(rnn_state)
                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions)  #obs[i][0:1]

                    rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),dtype=np.float32,)
                    masks = np.ones((self.n_rollout_threads, self.num_robots, 1), dtype=np.float32)
                    masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                    episode_rewards.append(rewards)

                elif method == 'orca':
                    obs = obs[0]
                    for i,o in enumerate(obs):
                        with open(agent_file[i], "a") as f:
                            line = f"{step+1}, {', '.join(map(str, o))}\n"
                            f.write(line)
                    
                    actions = self.orca_policy.predict(obs)  # (robot_num+human_num * 6)
                    obs = self.envs.step(actions)

                elif method == 'apf':
                    obs = obs[0]
                    actions = self.apf_policy.act(obs)
                    obs = self.envs.step(actions)

            episode_success = self.envs.render(mode=mode,visualize=visualize)
            
            if episode_success == True:
                success_times += 1
            success_rate = success_times / (episode + 1)
            print('success rate is:',success_rate)

            if method == 'ppo':
                episode_rewards = np.array(episode_rewards)
                for agent_id in range(self.num_robots):
                    average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                    print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        
