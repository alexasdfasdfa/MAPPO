import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from policy.mappo.actor_critic import R_Actor, R_Critic
from policy.utils.util import update_linear_schedule


class RMAPPOPolicy:
    """
    MAPPO Policy class. Wraps actor and critic networks to compute actions and value function predictions.
    """

    def __init__(self, args, robot_obs_space, human_obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.robot_obs_space = robot_obs_space
        self.human_obs_space = human_obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.robot_obs_space, self.human_obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # DDP support for multi-GPU training
        self.use_ddp = getattr(args, 'use_ddp', False)
        if self.use_ddp:
            # Use DistributedDataParallel (more efficient than DataParallel, no LSTM issues)
            local_rank = getattr(args, 'local_rank', 0)
            print("[DDP] Wrapping actor and critic with DistributedDataParallel on rank {}".format(local_rank))
            self.actor = DDP(self.actor, device_ids=[local_rank], output_device=local_rank)
            self.critic = DDP(self.critic, device_ids=[local_rank], output_device=local_rank)
        else:
            # Single GPU mode
            if torch.cuda.is_available():
                print("[Single GPU] Using device: {}".format(self.device))
            else:
                print("[CPU] Using CPU for training")

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, robot_obs, human_obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        # ========== 优化2: 优化数据传输 ==========
        robot_obs = self._to_tensor(robot_obs)
        human_obs = self._to_tensor(human_obs)
        cent_obs = self._to_tensor(cent_obs)
        rnn_states_actor = self._to_tensor(rnn_states_actor)
        rnn_states_critic = self._to_tensor(rnn_states_critic)
        masks = self._to_tensor(masks)
        
        if available_actions is not None:
            available_actions = self._to_tensor(available_actions)

        actions, action_log_probs, rnn_states_actor = self.actor(robot_obs,
                                                                 human_obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        cent_obs = self._to_tensor(cent_obs)
        rnn_states_critic = self._to_tensor(rnn_states_critic)
        masks = self._to_tensor(masks)
        
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, robot_obs, human_obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        # ========== 优化2: 优化数据传输 ==========
        robot_obs = self._to_tensor(robot_obs)
        human_obs = self._to_tensor(human_obs)
        cent_obs = self._to_tensor(cent_obs)
        rnn_states_actor = self._to_tensor(rnn_states_actor)
        rnn_states_critic = self._to_tensor(rnn_states_critic)
        action = self._to_tensor(action)
        masks = self._to_tensor(masks)
        
        if available_actions is not None:
            available_actions = self._to_tensor(available_actions)
        if active_masks is not None:
            active_masks = self._to_tensor(active_masks)

        if self.use_ddp:
            action_log_probs, dist_entropy = self.actor.module.evaluate_actions(robot_obs, 
                                                                                 human_obs,
                                                                                 rnn_states_actor,
                                                                                 action,
                                                                                 masks,
                                                                                 available_actions,
                                                                                 active_masks)
            values, _ = self.critic.module(cent_obs, rnn_states_critic, masks)
        else:
            action_log_probs, dist_entropy = self.actor.evaluate_actions(robot_obs, 
                                                                         human_obs,
                                                                         rnn_states_actor,
                                                                         action,
                                                                         masks,
                                                                         available_actions,
                                                                         active_masks)
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, robot_obs, human_obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        robot_obs = self._to_tensor(robot_obs)
        human_obs = self._to_tensor(human_obs)
        rnn_states_actor = self._to_tensor(rnn_states_actor)
        masks = self._to_tensor(masks)
        
        if available_actions is not None:
            available_actions = self._to_tensor(available_actions)
            
        actions, _, rnn_states_actor = self.actor(robot_obs, human_obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
