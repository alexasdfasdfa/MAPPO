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


class _CompatConsensusModule(nn.Module):
    """Name-compatible block for undet_v3_target_latent_global_rank checkpoints."""

    def __init__(self, *, p_max: int, d_h: int):
        super().__init__()
        self.p_max = int(p_max)
        in_dim = 2 * int(p_max)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, int(p_max) * 2),
        )

    def forward(self, neighbor_rel: torch.Tensor, neighbor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # neighbor_rel: (B, n=1, P, 2)
        B, n, P, _ = neighbor_rel.shape
        flat = neighbor_rel.reshape(B, n, P * 2)
        h = self.enc(flat)
        pred = self.dec(h).reshape(B, n, P, 2)
        return h, pred


class UndeterminedTargetHeadV3GlobalRankCompat(nn.Module):
    """
    MAPPO v3 head aligned to undet_v3_target_latent_global_rank checkpoint names:
    - consensus.*
    - consensus_global_goal_head.*
    - consensus_pred_goal_dist.*
    - consensus_pred_hun_disp.*

    This is a lightweight adapter over MAPPO robot_obs that preserves key structure
    so pretrained selector tensors can be loaded directly.
    """

    def __init__(self, args, m_slots: int):
        super().__init__()
        self.m = max(1, int(m_slots))
        self.d_h = int(getattr(args, "undet_v3_latent_d_h", 64))
        self.p_max = int(getattr(args, "undet_v3_latent_p_max_neighbors", 12))
        self.k_goals = self.m
        self.geo_in_dim = self.d_h + 2
        self.obs_goal_radius = float(getattr(args, "undetermined_obs_goal_radius", 5.0))

        self.consensus = _CompatConsensusModule(p_max=self.p_max, d_h=self.d_h)
        self.consensus_global_goal_head = nn.Linear(self.d_h, self.k_goals)
        mid = max(int(getattr(args, "undet_v3_latent_geo_head_hidden", 128)), self.d_h)
        self.consensus_pred_goal_dist = nn.Sequential(
            nn.Linear(self.geo_in_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, self.k_goals),
        )
        self.consensus_pred_hun_disp = nn.Sequential(
            nn.Linear(self.geo_in_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, 2),
        )

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        """
        robot_obs layout (v3 hybrid): [7 + 5*M + 1 + (v3 prev idx=1) + comm_tail + px,py]
        We build a single-agent proxy neighbor_rel from slot dx/dy to run consensus stack.
        """
        B = robot_obs.shape[0]
        M = self.m
        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        invalid = g[:, :, 4] < -0.05
        rel = g[:, :, :2] * self.obs_goal_radius  # recover approximate metric dx,dy

        # Build P-neighbor tensor expected by consensus: (B,1,P,2), pad/truncate to p_max.
        if M >= self.p_max:
            rel_p = rel[:, : self.p_max, :]
            mask_p = (~invalid[:, : self.p_max]).float()
        else:
            pad = rel.new_zeros(B, self.p_max - M, 2)
            rel_p = torch.cat([rel, pad], dim=1)
            pad_m = invalid.new_zeros(B, self.p_max - M)
            mask_p = (~torch.cat([invalid, pad_m], dim=1)).float()
        neighbor_rel = rel_p.unsqueeze(1)
        neighbor_mask = mask_p.unsqueeze(1)

        h, _pred_rel = self.consensus(neighbor_rel, neighbor_mask)  # (B,1,d_h)
        m = neighbor_mask.unsqueeze(-1)
        mean_rel = (m * neighbor_rel).sum(dim=-2) / m.sum(dim=-2).clamp(min=1e-6)  # (B,1,2)
        stem = torch.cat([h, mean_rel], dim=-1)  # (B,1,d_h+2)

        # Primary logits from compat global head; geo head adds a mild ranking prior.
        logits_head = self.consensus_global_goal_head(h).squeeze(1)  # (B,M)
        pred_goal_dist = self.consensus_pred_goal_dist(stem).squeeze(1)  # (B,M)
        logits = logits_head - pred_goal_dist
        logits = logits.masked_fill(invalid, -1e4)
        return logits

    def extract_p1_consensus(self, robot_obs: torch.Tensor) -> torch.Tensor:
        B = robot_obs.shape[0]
        M = self.m
        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        invalid = g[:, :, 4] < -0.05
        rel = g[:, :, :2] * self.obs_goal_radius
        if M >= self.p_max:
            rel_p = rel[:, : self.p_max, :]
            mask_p = (~invalid[:, : self.p_max]).float()
        else:
            pad = rel.new_zeros(B, self.p_max - M, 2)
            rel_p = torch.cat([rel, pad], dim=1)
            pad_m = invalid.new_zeros(B, self.p_max - M)
            mask_p = (~torch.cat([invalid, pad_m], dim=1)).float()
        neighbor_rel = rel_p.unsqueeze(1)
        neighbor_mask = mask_p.unsqueeze(1)
        h, _ = self.consensus(neighbor_rel, neighbor_mask)
        return h.squeeze(1)


class _CompatRecommenderGlobalStack(nn.Module):
    """Name-compatible recommender block for decoupled-rank checkpoints."""

    def __init__(self, *, d_h: int, e_key: int = 64, e_query: int = 64, pair_feat_dim: int = 6, geo_query_dim: int = 16):
        super().__init__()
        self.d_h = int(d_h)
        self.geo_query_dim = max(4, int(geo_query_dim))
        self.e_key = int(e_key)
        self.e_query = int(e_query)
        self.pair_fd = int(pair_feat_dim)
        self.scale = float(e_key) ** -0.5
        self.target_in = nn.Sequential(nn.Linear(self.pair_fd + d_h, e_key), nn.ReLU(), nn.Linear(e_key, e_key))
        self.dist_stat_compress = nn.Linear(4, self.geo_query_dim)
        q_in = 2 + self.d_h + 2 + self.geo_query_dim
        self.self_in = nn.Sequential(nn.Linear(q_in, e_query), nn.ReLU(), nn.Linear(e_query, e_query))
        self.q_align = nn.Linear(e_query, e_key) if e_query != e_key else nn.Identity()
        self.to_xy = nn.Sequential(nn.Linear(e_key, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(
        self,
        h: torch.Tensor,
        neighbor_rel: torch.Tensor,
        neighbor_mask: torch.Tensor,
        all_goal_feats: torch.Tensor,
        goal_xy: torch.Tensor,
        agent_xy: torch.Tensor,
        pred_goal_dist: torch.Tensor,
        pred_hun_disp: torch.Tensor,
        *,
        tau_logits: float = 0.35,
    ) -> dict[str, torch.Tensor]:
        B, n, K, fd = all_goal_feats.shape
        if fd != self.pair_fd:
            raise ValueError(f"all_goal_feats last dim {fd} != expected {self.pair_fd}")

        hb = h.unsqueeze(2).expand(B, n, K, -1)
        pairs = torch.cat([all_goal_feats, hb], dim=-1)
        e = self.target_in(pairs.reshape(B * n * K, -1)).reshape(B, n, K, -1)

        m = neighbor_mask.unsqueeze(-1)
        self_geo = (m * neighbor_rel).sum(dim=-2) / m.sum(dim=-2).clamp(min=1e-6)
        d_mean = pred_goal_dist.mean(dim=-1, keepdim=True)
        d_std = pred_goal_dist.std(dim=-1, keepdim=True).clamp_min(1e-6)
        d_min = pred_goal_dist.min(dim=-1, keepdim=True).values
        d_max = pred_goal_dist.max(dim=-1, keepdim=True).values
        d_stat = torch.cat([d_mean, d_std, d_min, d_max], dim=-1)
        d_q = torch.tanh(self.dist_stat_compress(d_stat))

        qv = self.self_in(torch.cat([self_geo, h, pred_hun_disp, d_q], dim=-1))
        q = self.q_align(qv).unsqueeze(2)
        logits = (e * q).sum(dim=-1) * self.scale

        t = max(float(tau_logits), 1e-6)
        p = torch.softmax(logits / t, dim=-1)
        g3 = goal_xy[:, None, :, :].expand(B, n, K, 2)
        y_soft = (p.unsqueeze(-1) * g3).sum(dim=-2)
        xy_hat = self.to_xy(e.reshape(B * n * K, -1)).reshape(B, n, K, 2)
        return {"logits": logits, "e_key": e, "p": p, "y_soft": y_soft, "per_goal_xy_hat": xy_hat}


class UndeterminedTargetHeadV3DecoupledRankCompat(nn.Module):
    """
    MAPPO v3 head aligned to undet_v3_target_latent_decoupled_rank checkpoint names:
    - consensus.*
    - consensus_pred_goal_dist_pair.*
    - consensus_pred_hun_disp.*
    - recommender.*
    """

    def __init__(self, args, m_slots: int):
        super().__init__()
        self.m = max(1, int(m_slots))
        self.d_h = int(getattr(args, "undet_v3_latent_d_h", 64))
        self.p_max = int(getattr(args, "undet_v3_latent_p_max_neighbors", 12))
        self.k_goals = self.m
        self.obs_goal_radius = float(getattr(args, "undetermined_obs_goal_radius", 5.0))
        self.pair_feat_dim = 6
        self.geo_in_dim = self.d_h + 2

        self.consensus = _CompatConsensusModule(p_max=self.p_max, d_h=self.d_h)
        mid = max(int(getattr(args, "undet_v3_latent_geo_head_hidden", 128)), self.d_h)
        self.consensus_pred_goal_dist_pair = nn.Sequential(
            nn.Linear(self.geo_in_dim + self.pair_feat_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )
        self.consensus_pred_hun_disp = nn.Sequential(
            nn.Linear(self.geo_in_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, 2),
        )
        self.recommender = _CompatRecommenderGlobalStack(
            d_h=self.d_h,
            e_key=int(getattr(args, "undet_v3_decoupled_e_key", 64)),
            e_query=int(getattr(args, "undet_v3_decoupled_e_query", 64)),
            pair_feat_dim=self.pair_feat_dim,
            geo_query_dim=int(getattr(args, "undet_v3_decoupled_geo_query_dim", 16)),
        )
        self.tau_logits = float(getattr(args, "undet_v3_decoupled_tau_logits", 0.35))

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        B = robot_obs.shape[0]
        M = self.m
        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        invalid = g[:, :, 4] < -0.05
        rel = g[:, :, :2] * self.obs_goal_radius

        if M >= self.p_max:
            rel_p = rel[:, : self.p_max, :]
            mask_p = (~invalid[:, : self.p_max]).float()
        else:
            pad = rel.new_zeros(B, self.p_max - M, 2)
            rel_p = torch.cat([rel, pad], dim=1)
            pad_m = invalid.new_zeros(B, self.p_max - M)
            mask_p = (~torch.cat([invalid, pad_m], dim=1)).float()

        neighbor_rel = rel_p.unsqueeze(1)
        neighbor_mask = mask_p.unsqueeze(1)
        h, _ = self.consensus(neighbor_rel, neighbor_mask)
        m = neighbor_mask.unsqueeze(-1)
        mean_rel = (m * neighbor_rel).sum(dim=-2) / m.sum(dim=-2).clamp(min=1e-6)
        stem = torch.cat([h, mean_rel], dim=-1)

        raw_dx = rel[:, :, 0]
        raw_dy = rel[:, :, 1]
        dist = torch.sqrt(raw_dx.pow(2) + raw_dy.pow(2) + 1e-8)
        pair = torch.stack([raw_dx, raw_dy, g[:, :, 2], g[:, :, 3], g[:, :, 4], dist], dim=-1).unsqueeze(1)
        stem_b = stem.unsqueeze(2).expand(B, 1, M, self.geo_in_dim)
        pair_geo = torch.cat([stem_b, pair], dim=-1)
        pred_goal_dist = self.consensus_pred_goal_dist_pair(pair_geo.reshape(B * M, -1)).reshape(B, 1, M)
        pred_hun_disp = self.consensus_pred_hun_disp(stem)

        goal_xy = rel
        agent_xy = robot_obs[:, -2:].unsqueeze(1)
        out = self.recommender(
            h,
            neighbor_rel,
            neighbor_mask,
            pair,
            goal_xy,
            agent_xy,
            pred_goal_dist,
            pred_hun_disp,
            tau_logits=self.tau_logits,
        )
        logits = out["logits"].squeeze(1)
        logits = logits.masked_fill(invalid, -1e4)
        return logits

    def extract_p1_consensus(self, robot_obs: torch.Tensor) -> torch.Tensor:
        B = robot_obs.shape[0]
        M = self.m
        g = robot_obs[:, 7 : 7 + 5 * M].reshape(B, M, 5)
        invalid = g[:, :, 4] < -0.05
        rel = g[:, :, :2] * self.obs_goal_radius
        if M >= self.p_max:
            rel_p = rel[:, : self.p_max, :]
            mask_p = (~invalid[:, : self.p_max]).float()
        else:
            pad = rel.new_zeros(B, self.p_max - M, 2)
            rel_p = torch.cat([rel, pad], dim=1)
            pad_m = invalid.new_zeros(B, self.p_max - M)
            mask_p = (~torch.cat([invalid, pad_m], dim=1)).float()
        neighbor_rel = rel_p.unsqueeze(1)
        neighbor_mask = mask_p.unsqueeze(1)
        h, _ = self.consensus(neighbor_rel, neighbor_mask)
        return h.squeeze(1)
