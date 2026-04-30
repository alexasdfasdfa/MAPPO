import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.mappo.cons_decaf_utils import kl_ce_loss_paper
from policy.mappo.undet_v2_latent_ckpt import undetermined_v2_slot_supervision_loss
from policy.utils.util import get_gard_norm, huber_loss, mse_loss
from policy.utils.valuenorm import ValueNorm
from policy.mappo.utils.util import check
import time


class RMAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self._args = args
        self._undet_v2_head_aux_coef = float(getattr(args, "undet_v2_target_head_aux_coef", 0.0))
        self._undet_v2_head_aux_on = (
            self._undet_v2_head_aux_coef > 0.0
            and getattr(args, "enable_undetermined_goal_v2", False)
            and not getattr(args, "use_attn_comm_actor", False)
            and str(getattr(args, "undet_v2_latent_train_mode", "finetune_all")) == "finetune_all"
            and getattr(policy.actor, "undetermined_head", None) is not None
        )
        self._undet_v3_kl_coef = float(getattr(args, "undetermined_v3_target_kl_coef", 0.0))

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self._train_progress = 0.0

    def set_training_progress(self, progress: float) -> None:
        try:
            p = float(progress)
        except (TypeError, ValueError):
            p = 0.0
        self._train_progress = min(1.0, max(0.0, p))

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 15:
            (
                share_obs_batch,
                robot_obs_batch,
                human_obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                comm_rnn_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                old_undet_target_logits_batch,
            ) = sample
        else:
            (
                share_obs_batch,
                robot_obs_batch,
                human_obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                comm_rnn_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            ) = sample
            old_undet_target_logits_batch = None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        comm_arg = None if comm_rnn_states_batch is None else check(comm_rnn_states_batch).to(**self.tpdv)
        self.policy.actor_optimizer.zero_grad()
        if getattr(self.policy, "ce_optimizer", None) is not None:
            self.policy.ce_optimizer.zero_grad()
        self.policy.critic_optimizer.zero_grad()

        values, action_log_probs, dist_entropy, logits_hat, out_ctx, flat_logits = self.policy.evaluate_actions(
            share_obs_batch,
            robot_obs_batch,
            human_obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            comm_rnn_states_actor=comm_arg,
        )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        cc = float(getattr(self.policy, "cons_mac_ce_coef", 0.0))
        dc = float(getattr(self.policy, "cons_mac_distill_coef", 0.0))
        t_out = None
        t_flat = None
        if dc > 0.0 and getattr(self.policy, "teacher_actor", None) is not None and flat_logits is not None:
            with torch.no_grad():
                _, _, _, _, t_out, t_flat = self.policy.teacher_actor.evaluate_actions(
                    robot_obs_batch,
                    human_obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    available_actions_batch,
                    active_masks_batch,
                    comm_rnn_states=comm_arg,
                )

        loss_actor = policy_loss - dist_entropy * self.entropy_coef
        if (
            dc > 0.0
            and t_flat is not None
            and flat_logits is not None
            and t_flat.shape == flat_logits.shape
        ):
            loss_actor = loss_actor + dc * F.mse_loss(flat_logits, t_flat)

        undet_head_aux = None
        if update_actor and self._undet_v2_head_aux_on:
            ro = check(robot_obs_batch).to(**self.tpdv)
            logits_u = self.policy.actor.get_undetermined_target_logits(ro)
            rr = float(getattr(self._args, "undetermined_obs_goal_radius", 5.0))
            am = active_masks_batch if self._use_policy_active_masks else None
            aux = undetermined_v2_slot_supervision_loss(ro, logits_u, goal_rr=rr, active_mask=am)
            undet_head_aux = self._undet_v2_head_aux_coef * aux
            loss_actor = loss_actor + undet_head_aux

        undet_v3_kl = None
        _kl_coef = float(self._undet_v3_kl_coef)
        if bool(getattr(self._args, "undetermined_v3_curriculum_enable", True)):
            ratio = float(getattr(self._args, "undetermined_v3_curriculum_motion_phase_ratio", 0.45))
            if self._train_progress < ratio:
                _kl_coef *= float(getattr(self._args, "undetermined_v3_curriculum_selector_kl_scale_early", 0.25))
            else:
                _kl_coef *= float(getattr(self._args, "undetermined_v3_curriculum_selector_kl_scale_late", 1.50))
        if (
            update_actor
            and _kl_coef > 1e-12
            and old_undet_target_logits_batch is not None
            and getattr(self._args, "enable_undetermined_goal_v3", False)
        ):
            ro = check(robot_obs_batch).to(**self.tpdv)
            new_logits = self.policy.actor.get_undetermined_target_logits(ro)
            old_l = check(old_undet_target_logits_batch).to(**self.tpdv).detach()
            if new_logits.shape == old_l.shape:
                p_new = torch.softmax(new_logits, dim=-1).clamp_min(1e-8)
                log_p_new = torch.log(p_new)
                p_old = torch.softmax(old_l, dim=-1).clamp_min(1e-8)
                kl = (p_old * (torch.log(p_old) - log_p_new)).sum(dim=-1, keepdim=True)
                if self._use_policy_active_masks:
                    undet_v3_kl = (kl * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    undet_v3_kl = kl.mean()
                loss_actor = loss_actor + _kl_coef * undet_v3_kl

        l_ce_side = None
        if getattr(self.policy, "ce_optimizer", None) is not None:
            ce_parts = []
            if (
                cc > 0.0
                and logits_hat is not None
                and getattr(self.policy, "global_label_head", None) is not None
            ):
                share_flat = check(share_obs_batch).to(**self.tpdv).reshape(logits_hat.shape[0], -1)
                e_g = self.policy.global_label_head(share_flat).detach()
                if e_g.shape[-1] == logits_hat.shape[-1]:
                    ce_parts.append(cc * kl_ce_loss_paper(e_g, logits_hat))
            if (
                dc > 0.0
                and out_ctx is not None
                and t_out is not None
                and out_ctx.shape == t_out.shape
            ):
                ce_parts.append(dc * F.mse_loss(out_ctx, t_out))
            if ce_parts:
                l_ce_side = sum(ce_parts)

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        need_retain_ce = update_actor and (l_ce_side is not None)
        if update_actor:
            loss_actor.backward(retain_graph=need_retain_ce)
            if l_ce_side is not None:
                l_ce_side.backward(retain_graph=True)

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        if update_actor:
            self.policy.actor_optimizer.step()

        if update_actor and getattr(self.policy, "ce_optimizer", None) is not None and l_ce_side is not None:
            if self._use_max_grad_norm:
                nn.utils.clip_grad_norm_(
                    [p for g in self.policy.ce_optimizer.param_groups for p in g["params"]],
                    self.max_grad_norm,
                )
            self.policy.ce_optimizer.step()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        undet_aux_item = float(undet_head_aux.detach().item()) if undet_head_aux is not None else None
        undet_v3_kl_item = float(undet_v3_kl.detach().item()) if undet_v3_kl is not None else None

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
            undet_aux_item,
            undet_v3_kl_item,
        )

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        if self._undet_v2_head_aux_on:
            train_info["undet_head_aux_loss"] = 0.0
        if self._undet_v3_kl_coef > 1e-12 and getattr(self._args, "enable_undetermined_goal_v3", False):
            train_info["undet_v3_target_kl"] = 0.0

        for _ in range(self.ppo_epoch):#耗时16
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)  #用yield在函数中返回可迭代的结果

            for sample in data_generator:#每轮耗时约1s
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                    undet_aux_item,
                    undet_v3_kl_item,
                ) = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                if self._undet_v2_head_aux_on and undet_aux_item is not None:
                    train_info["undet_head_aux_loss"] += undet_aux_item
                if (
                    self._undet_v3_kl_coef > 1e-12
                    and getattr(self._args, "enable_undetermined_goal_v3", False)
                    and undet_v3_kl_item is not None
                ):
                    train_info["undet_v3_target_kl"] += undet_v3_kl_item

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
