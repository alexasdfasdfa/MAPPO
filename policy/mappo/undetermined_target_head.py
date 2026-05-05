"""
Target preference head for undetermined-goal mode: ego embedding dot goal embeddings.
Obs layout must match env_core._pack_undetermined_obs (px,py last two).

``UndeterminedTargetHeadV2PairMLP`` matches ``undet_v2_target_latent.TargetLatentSelector`` (pair MLP + score_dir)
so weights from ``target_latent_selector.pt`` can load under ``undetermined_head.*``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UndeterminedTargetHeadV2PairMLP(nn.Module):
    """
    Same tensor ops as ``undet_v2_target_latent.model.TargetLatentSelector`` for one decentralized row:
    full v2 obs (incl. px,py) is repeated per slot; pair = [obs | rel_x/cs | rel_y/cs | k_norm] with rel from packed dx,dy.

    Requires pure v2 packing (no AttnComm hybrid tail): ``robot_obs_dim == 7 + 5*M + 1`` before px,py.
    """

    def __init__(self, args, m_slots: int):
        super().__init__()
        self._args = args
        self.m = max(1, int(m_slots))
        self.pair_extra = 3
        core = 7 + 5 * self.m + 1
        self.obs_d = core + 2
        rod = int(getattr(args, "robot_obs_dim", core))
        if rod != core:
            raise ValueError(
                f"UndeterminedTargetHeadV2PairMLP needs pure undetermined v2 obs: robot_obs_dim={rod} "
                f"but expected {core} (7+5*M+1). Disable hybrid AttnComm tail or use dot_product head."
            )
        self.hidden = int(getattr(args, "undet_v2_pair_mlp_hidden", 384))
        self.d_emb = int(getattr(args, "undetermined_target_embed_dim", 96))
        use_ln = not bool(getattr(args, "undet_v2_pair_mlp_no_layernorm", False))
        in_d = self.obs_d + self.pair_extra
        if use_ln:
            self.mlp = nn.Sequential(
                nn.Linear(in_d, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.d_emb),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_d, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.d_emb),
            )
        self.score_dir = nn.Parameter(torch.randn(self.d_emb) * 0.02)
        self.raw_dist_scale = nn.Parameter(torch.tensor(1.2))

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        B = robot_obs.shape[0]
        M = self.m
        obs_d = self.obs_d
        robot_obs = robot_obs[:, :obs_d]
        rr = float(getattr(self._args, "undetermined_obs_goal_radius", 8.0))
        scale_env = max(rr, 1.0)
        coord_scale = max(float(getattr(self._args, "undet_v2_pair_coord_scale", 10.0)), 1e-6)
        box = float(getattr(self._args, "undet_v2_pair_dist_box", 10.0))
        box_sq = box * box

        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        invalid = g[:, :, 4] < -0.05

        obs_e = robot_obs.unsqueeze(1).expand(B, M, obs_d)
        raw_dx = g[:, :, 0] * scale_env
        raw_dy = g[:, :, 1] * scale_env
        rel_cs = torch.stack([raw_dx / coord_scale, raw_dy / coord_scale], dim=-1)
        k_norm = g[:, :, 4:5]
        pair_in = torch.cat([obs_e, rel_cs, k_norm], dim=-1)
        h = self.mlp(pair_in.reshape(B * M, -1)).view(B, M, -1)
        logits = (h * self.score_dir.view(1, 1, -1)).sum(dim=-1)
        dist_sq = raw_dx.pow(2) + raw_dy.pow(2)
        gate = F.softplus(self.raw_dist_scale)
        logits = logits - gate * (dist_sq / max(box_sq, 1e-8))
        logits = logits.masked_fill(invalid, -1e4)
        return logits


class UndeterminedTargetHeadV3Attention(nn.Module):
    """
    v3: scaled dot-product attention over M nearest-goal slot embeddings (invalid slots masked to ~0 weight),
    plus a per-slot fusion MLP on (slot, pooled context, query, ego+pend+prev) as a residual on attention logits.

    Obs layout matches env_core._pack_undetermined_v3_obs (px, py last two):
      slots: robot_obs[:, 7 : 7 + 5*M], pend: [7+5M : 7+5M+1], prev_tid_norm: [7+5M+1 : 7+5M+2], then AttnComm tail, then px,py.
    """

    def __init__(self, args, m_slots: int):
        super().__init__()
        self.m = max(1, int(m_slots))
        E = max(8, int(getattr(args, "undetermined_v3_attn_embed_dim", 64)))
        self._E = E
        self.scale = float(E) ** -0.5
        H = max(8, int(getattr(args, "undetermined_v3_fuse_hidden", 128)))
        self.slot_proj = nn.Sequential(
            nn.Linear(5, E),
            nn.ReLU(),
            nn.Linear(E, E),
        )
        # ego 7 + pend + prev scalar
        self.q_mlp = nn.Sequential(
            nn.Linear(7 + 1 + 1, E),
            nn.ReLU(),
            nn.Linear(E, E),
        )
        # slot_emb, ctx_broadcast, q_broadcast, ego_bundle (9)
        self.delta_mlp = nn.Sequential(
            nn.Linear(E + E + E + 9, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        B = robot_obs.shape[0]
        M = self.m
        E = self._E
        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        pend = robot_obs[:, 7 + 5 * M : 7 + 5 * M + 1]
        prev = robot_obs[:, 7 + 5 * M + 1 : 7 + 5 * M + 2]
        invalid = g[:, :, 4] < -0.05

        slot_emb = self.slot_proj(g.reshape(B * M, 5)).reshape(B, M, E)
        ego_bundle = torch.cat([robot_obs[:, :7], pend, prev], dim=-1)
        q = self.q_mlp(ego_bundle)
        attn_logits = (slot_emb * q.unsqueeze(1)).sum(-1) * self.scale
        attn_logits = attn_logits.masked_fill(invalid, -1e4)
        w = torch.softmax(attn_logits, dim=-1)
        ctx = (w.unsqueeze(-1) * slot_emb).sum(dim=1)
        ctx_e = ctx.unsqueeze(1).expand(B, M, E)
        q_e = q.unsqueeze(1).expand(B, M, E)
        ego_e = ego_bundle.unsqueeze(1).expand(B, M, 9)
        fusion_in = torch.cat([slot_emb, ctx_e, q_e, ego_e], dim=-1)
        delta = self.delta_mlp(fusion_in.reshape(B * M, -1)).reshape(B, M)
        logits = attn_logits + delta
        logits = logits.masked_fill(invalid, -1e4)
        return logits
