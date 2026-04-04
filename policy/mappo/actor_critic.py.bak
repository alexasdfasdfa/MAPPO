import torch
import torch.nn as nn
import numpy as np
from policy.mappo.utils.util import init, check
from policy.mappo.utils.cnn import CNNBase
from policy.mappo.utils.mlp import MLPBase
from policy.mappo.utils.rnn import RNNLayer
from policy.mappo.utils.act import ACTLayer
from policy.mappo.utils.lstm import LSTMLayer
from policy.mappo.utils.popart import PopArt
from policy.utils.util import get_shape_from_obs_space
from policy.mappo.utils.util import transform
import time


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, robot_obs_space, human_obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.att_agents = args.num_attention_agents
        self.human_num = args.num_humans
        self.use_human_obs = getattr(args, "use_human_obs", True)

        self.robot_obs_shape = args.robot_obs_dim + 2
        self.human_obs_shape = args.human_obs_dim

        base = MLPBase
        self.base_robot = base(args, self.robot_obs_shape)
        self.base_human = base(args) # Used to make the order of magnitude of both robot_feature and human_feature

        self.lstm = LSTMLayer(self.human_obs_shape, self.hidden_size, self._recurrent_N)

        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size*2, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, robot_obs, human_obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.[batch, obs_dim]
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        robot_obs = check(robot_obs).to(**self.tpdv)
        # human_obs = check(human_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if self.use_human_obs:
            human_obs = transform(robot_obs, human_obs, self.human_num, self.att_agents)
            human_features = self.lstm(human_obs)
            human_features = self.base_human(human_features)
        else:
            # when human observations are disabled, use a zero vector as the human branch feature
            human_features = torch.zeros((robot_obs.shape[0], self.hidden_size), device=robot_obs.device, dtype=robot_obs.dtype)

        robot_obs = robot_obs[:,:self.robot_obs_shape]
        actor_features = self.base_robot(robot_obs)
        total_features = torch.cat((actor_features,human_features), dim=1)  #[thread 128]

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(total_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, robot_obs, human_obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        robot_obs = check(robot_obs).to(**self.tpdv)
        # human_obs = check(human_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.use_human_obs:
            human_obs = transform(robot_obs, human_obs, self.human_num, self.att_agents)
            human_features = self.lstm(human_obs)
            human_features = self.base_human(human_features)
        else:
            human_features = torch.zeros((robot_obs.shape[0], self.hidden_size), device=robot_obs.device, dtype=robot_obs.dtype)

        robot_obs = robot_obs[:,:self.robot_obs_shape]
        actor_features = self.base_robot(robot_obs)
        total_features = torch.cat((actor_features,human_features), dim=1)

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(total_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.human_num = args.num_humans
        self.robot_num = args.num_agents
        self.robot_obs_dim = args.robot_obs_dim
        self.human_obs_dim = args.human_obs_dim
        self.att_agents = args.num_attention_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        # neighbourhood / agent-state selection config
        self.agent_state_mode = getattr(args, "agent_state_mode", "all")
        self.neighbor_n = getattr(args, "neighbor_n", 5)
        self.neighbor_radius = getattr(args, "neighbor_radius", 5.0)
        self.neighbor_distance_metric = getattr(args, "neighbor_distance_metric", "euclidean")
        self.neighbor_padding_value = getattr(args, "neighbor_padding_value", 0.0)

        # per-row obs layout from env_core: (1 + max(num_humans, num_attention_agents), obs_dim + 2)
        # where obs_dim = max(robot_obs_dim, human_obs_dim)
        self.obs_row_dim = max(self.robot_obs_dim, self.human_obs_dim) + 2
        self.num_rows = 1 + max(self.human_num, self.att_agents)
        # robot feature vector (includes px, py as last two entries)
        self.agent_feat_dim = self.robot_obs_dim + 2

        # how many agent-state rows are fed into critic MLP
        if self.agent_state_mode == "all":
            self.num_agent_features = self.robot_num
        else:
            self.num_agent_features = self.neighbor_n

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # critic input dimension = num_agent_features * agent_feat_dim
        cent_obs_shape = self.num_agent_features * self.agent_feat_dim
        base = MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)

        # cent_obs comes from SharedReplayBuffer.share_obs, which stores for each
        # (env, agent) a flattened joint observation of ALL robots + humans:
        #   share_obs_dim = robot_num * num_rows * obs_row_dim
        # For a fixed env, each agent's cent_obs row is identical.
        batch = cent_obs.shape[0]
        cent_obs = cent_obs.view(batch, -1)

        # infer number of envs in this batch (batch = n_envs * robot_num)
        assert batch % self.robot_num == 0, "Batch size must be a multiple of robot_num."
        n_envs = batch // self.robot_num

        # we will build neighbourhood features per (env, self_agent) sample
        device = cent_obs.device
        neighbour_feats = torch.empty(
            batch, self.num_agent_features, self.agent_feat_dim, device=device, dtype=cent_obs.dtype
        )

        # loop over envs and agents to build neighbourhoods
        for env_id in range(n_envs):
            # take one agent's centralized obs as the joint state for this env
            joint = cent_obs[env_id * self.robot_num]  # (share_obs_dim,)
            # guard against any length mismatch by truncating to the expected size
            expected_dim = self.robot_num * self.num_rows * self.obs_row_dim
            joint = joint[:expected_dim]
            joint = joint.view(self.robot_num, self.num_rows, self.obs_row_dim)

            # per-robot features are row 0, last two entries are (px, py)
            env_robots = joint[:, 0, : self.agent_feat_dim]  # (robot_num, agent_feat_dim)
            positions = env_robots[:, -2:]  # (robot_num, 2)

            for self_id in range(self.robot_num):
                global_idx = env_id * self.robot_num + self_id
                self_pos = positions[self_id]  # (2,)
                diffs = positions - self_pos  # (robot_num, 2)
                if self.neighbor_distance_metric == "manhattan":
                    dists = diffs.abs().sum(dim=-1)
                else:
                    dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8)

                if self.agent_state_mode == "all":
                    # use all robots (including self)
                    order = torch.argsort(dists)
                    selected_idx = order[: self.num_agent_features]
                else:
                    # nearest_n / nearest_n_radius
                    order = torch.argsort(dists)
                    k = min(self.neighbor_n, self.robot_num)
                    if self.agent_state_mode == "nearest_n":
                        selected_idx = order[:k]
                    else:  # "nearest_n_radius"
                        within = order[dists[order] <= self.neighbor_radius]
                        selected_idx = within[:k]

                selected = env_robots[selected_idx]  # (m, agent_feat_dim)

                # padding / trimming to fixed num_agent_features
                m = selected.shape[0]
                if m < self.num_agent_features:
                    pad = torch.full(
                        (self.num_agent_features - m, self.agent_feat_dim),
                        self.neighbor_padding_value,
                        device=device,
                        dtype=env_robots.dtype,
                    )
                    selected = torch.cat([selected, pad], dim=0)
                elif m > self.num_agent_features:
                    selected = selected[: self.num_agent_features]

                neighbour_feats[global_idx] = selected

        critic_input = neighbour_feats.view(batch, -1)
        critic_features = self.base(critic_input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
