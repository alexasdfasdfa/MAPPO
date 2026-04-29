import sys
import time
import torch
from policy.mappo.actor_critic import R_Actor, R_Critic
from policy.mappo.cons_decaf_utils import FrozenGlobalLabelProjector
from policy.utils.util import update_linear_schedule
from policy.utils.util import get_shape_from_obs_space, compute_joint_share_obs_flat_dim


def _undet_motion_only_freeze_head(args) -> bool:
    lat_v2 = getattr(args, "undet_v2_target_latent_model_dir", None)
    lat_v3 = getattr(args, "undet_v3_target_latent_model_dir", None)
    if lat_v2 and str(lat_v2).strip():
        if str(getattr(args, "undet_v2_latent_train_mode", "finetune_all")) != "motion_only":
            return False
        if not getattr(args, "enable_undetermined_goal_v2", False):
            return False
        if getattr(args, "use_attn_comm_actor", False):
            return False
        return True
    if lat_v3 and str(lat_v3).strip():
        if str(getattr(args, "undet_v3_latent_train_mode", "finetune_all")) != "motion_only":
            return False
        if not getattr(args, "enable_undetermined_goal_v3", False):
            return False
        return True
    return False


def _cons_mac_ce_param_filter(name: str) -> bool:
    """CE / ConsMAC stack (Ψ): exclude PE heads that only consume detached consensus (fuse, msg, human, obst)."""
    if not name.startswith("attn_comm_encoder."):
        return False
    for bad in ("fuse.", "msg_head.", "embed_human.", "attn_human.", "norm_human.", "embed_obst."):
        if bad in name:
            return False
    return True


class RMAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
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
        self._undet_goal_k = int(args.num_agents)

        self._is_cons_decaf = str(getattr(args, "architecture_mode", "default")) == "attn_undetermined_goal"
        self.cons_mac_ce_coef = float(getattr(args, "cons_mac_ce_coef", 0.0))
        self.cons_mac_distill_coef = float(getattr(args, "cons_mac_distill_coef", 0.0))
        _ce_lr = float(getattr(args, "cons_mac_ce_lr", self.lr))
        self._cons_mac_ce_lr = self.lr if _ce_lr < 0.0 else _ce_lr
        _gym_cent_flat = int(get_shape_from_obs_space(cent_obs_space)[0])
        self._share_obs_flat_dim = compute_joint_share_obs_flat_dim(args)
        if _gym_cent_flat != self._share_obs_flat_dim:
            print(
                f"[RMAPPOPolicy] cent_obs_space flat dim ({_gym_cent_flat}) != "
                f"buffer joint share_obs dim ({self._share_obs_flat_dim}); "
                "using buffer dim for global_label_head / CE."
            )

        ce_params = []
        pe_params = []
        _freeze_uh = _undet_motion_only_freeze_head(args)
        for name, p in self.actor.named_parameters():
            if not p.requires_grad:
                continue
            if _freeze_uh and name.startswith("undetermined_head."):
                p.requires_grad = False
                continue
            if self._is_cons_decaf and self.actor.use_attn_comm and _cons_mac_ce_param_filter(name):
                ce_params.append(p)
            else:
                pe_params.append(p)

        if _freeze_uh:
            print(
                "[RMAPPOPolicy] undetermined latent train_mode=motion_only: undetermined_head params frozen "
                "(not in actor optimizer)."
            )

        self.actor_optimizer = torch.optim.Adam(
            pe_params,
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        tdir = getattr(args, "cons_mac_teacher_model_dir", None)
        _has_teacher = bool(tdir and str(tdir).strip())
        self.ce_optimizer = None
        if (
            self._is_cons_decaf
            and self.actor.use_attn_comm
            and len(ce_params) > 0
            and (
                self.cons_mac_ce_coef > 0.0
                or (self.cons_mac_distill_coef > 0.0 and _has_teacher)
            )
        ):
            self.ce_optimizer = torch.optim.Adam(
                ce_params,
                lr=self._cons_mac_ce_lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
        self.global_label_head = None
        if self._is_cons_decaf and self.actor.use_attn_comm and self.cons_mac_ce_coef > 0.0:
            k = max(2, int(getattr(args, "cons_mac_ce_bins", 32)))
            self.global_label_head = FrozenGlobalLabelProjector(self._share_obs_flat_dim, k).to(self.device)

        self.teacher_actor = None
        if _has_teacher:
            self.teacher_actor = R_Actor(args, self.robot_obs_space, self.human_obs_space, self.act_space, self.device)
            sd = torch.load(str(tdir), map_location=self.device)
            self.teacher_actor.load_state_dict(sd, strict=False)
            self.teacher_actor.eval()
            for p in self.teacher_actor.parameters():
                p.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        if self.ce_optimizer is not None:
            update_linear_schedule(self.ce_optimizer, episode, episodes, self._cons_mac_ce_lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        cent_obs,
        robot_obs,
        human_obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        comm_rnn_states_actor=None,
    ):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor, comm_rnn_out, broadcast_msg = self.actor(
            robot_obs,
            human_obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            comm_rnn_states=comm_rnn_states_actor,
        )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, comm_rnn_out, broadcast_msg

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        cent_obs,
        robot_obs,
        human_obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        comm_rnn_states_actor=None,
    ):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        #耗时约1s
        action_log_probs, dist_entropy, logits_hat, out_ctx, flat_logits = self.actor.evaluate_actions(
            robot_obs,
            human_obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
            comm_rnn_states=comm_rnn_states_actor,
        )

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, logits_hat, out_ctx, flat_logits

    def act(self, robot_obs, human_obs, rnn_states_actor, masks, available_actions=None, deterministic=False,
            comm_rnn_states_actor=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor, _, _ = self.actor(
            robot_obs,
            human_obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            comm_rnn_states=comm_rnn_states_actor,
        )
        return actions, rnn_states_actor

    @torch.no_grad()
    def get_undetermined_targets(self, robot_obs, deterministic=False):
        logits = self.actor.get_undetermined_target_logits(robot_obs)
        if getattr(self.actor, "enable_undetermined_v2", False):
            return self._undetermined_v2_slot_to_global_tid(robot_obs, logits, deterministic)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def _undetermined_v2_slot_to_global_tid(self, robot_obs, logits, deterministic):
        """Map categorical over M nearest slots to global goal indices in [0, K-1]."""
        ro = torch.as_tensor(robot_obs, dtype=torch.float32, device=logits.device)
        M = logits.shape[1]
        K = max(1, int(self._undet_goal_k))
        g = ro[:, 7 : 7 + 5 * M].reshape(-1, M, 5)
        if deterministic:
            slot = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            slot = dist.sample()
        b_idx = torch.arange(ro.shape[0], device=ro.device, dtype=torch.long)
        k_norm = g[b_idx, slot.long(), 4]
        scale = float(max(K - 1, 1))
        tid = (k_norm * scale).round().long().clamp(0, K - 1)
        return tid

    @torch.no_grad()
    def get_v3_exchange_choices(self, robot_obs, deterministic=False):
        logits = self.actor.get_v3_exchange_logits(robot_obs)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()