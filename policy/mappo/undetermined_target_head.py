"""
Target preference head for undetermined-goal mode: ego embedding dot goal embeddings.
Obs layout must match env_core._pack_undetermined_obs (px,py last two).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UndeterminedTargetHead(nn.Module):
    def __init__(self, args, k_goals: int):
        super().__init__()
        self.k = int(k_goals)
        self.d = int(getattr(args, "undetermined_target_embed_dim", 32))
        self.goal_mlp = nn.Sequential(
            nn.Linear(4, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.ego_mlp = nn.Sequential(
            nn.Linear(8, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        """
        :param robot_obs: (B, robot_obs_dim + 2) with px, py last
        :return: logits (B, K)
        """
        B = robot_obs.shape[0]
        K = self.k
        g = robot_obs[:, 7 : 7 + 4 * K].reshape(B, K, 4)
        off = 7 + 4 * K
        pend = robot_obs[:, off + max(0, K - 1) : off + max(0, K - 1) + 1]
        ego_in = torch.cat([robot_obs[:, :7], pend], dim=-1)
        ego_e = self.ego_mlp(ego_in)
        ge = self.goal_mlp(g.reshape(B * K, 4)).reshape(B, K, self.d)
        logits = (ego_e.unsqueeze(1) * ge).sum(-1)
        return logits


class UndeterminedTargetHeadV2(nn.Module):
    """
    v2: M nearest-goal slots, 5-D per slot (dx, dy, in_r, cobs, k_norm). Logits over slots; env maps slot -> global tid.
    Obs layout must match env_core._pack_undetermined_v2_obs (px, py last two).
    """

    def __init__(self, args, m_slots: int):
        super().__init__()
        self.m = max(1, int(m_slots))
        self.d = int(getattr(args, "undetermined_target_embed_dim", 32))
        self.goal_mlp = nn.Sequential(
            nn.Linear(5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.ego_mlp = nn.Sequential(
            nn.Linear(8, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        """
        :param robot_obs: (B, robot_obs_dim + 2) with px, py last
        :return: logits (B, M) over nearest-goal slots (invalid slots should be masked by caller if needed)
        """
        B = robot_obs.shape[0]
        M = self.m
        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        pend = robot_obs[:, 7 + 5 * M : 7 + 5 * M + 1]
        ego_in = torch.cat([robot_obs[:, :7], pend], dim=-1)
        ego_e = self.ego_mlp(ego_in)
        ge = self.goal_mlp(g.reshape(B * M, 5)).reshape(B, M, self.d)
        logits = (ego_e.unsqueeze(1) * ge).sum(-1)
        invalid = g[:, :, 4] < -0.05
        logits = logits.masked_fill(invalid, -1e4)
        return logits
