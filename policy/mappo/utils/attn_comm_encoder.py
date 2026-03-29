"""
Grouped observation encoders + cross-step intent communication for MAPPO actor (fixed targets).

- Obs includes per-ally-slot **last-step broadcast** vectors from identified teammates.
- Slot features fuse geometry with received messages; GRU carries long-term comm memory.
- Outgoing broadcast = attention over {ego, slots} mixing self state and received neighbor info.
- Internal width `d` (attn_comm_hidden_dim) can be < hidden_size to save params/VRAM; fuse still maps to hidden_size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _n_heads_for(width: int, preferred: int) -> int:
    h = max(1, int(preferred))
    while h > 1 and width % h != 0:
        h -= 1
    return h


class AttnCommActorEncoder(nn.Module):
    """
    robot_obs layout (px, py last):
      [self_7 | allies P*6 | recv P*M | humans H*5 | obstacle_4 | px py]
    """

    SELF_DIM = 7
    ALLY_DIM = 6
    HUMAN_DIM = 5
    OBST_DIM = 4

    def __init__(self, args):
        super().__init__()
        self.hidden_size = int(args.hidden_size)
        # Internal trunk width; 0 or None -> use hidden_size (legacy full-width)
        _hd = getattr(args, "attn_comm_hidden_dim", None)
        self.d = int(_hd) if _hd is not None and int(_hd) > 0 else self.hidden_size
        self.P = max(0, int(getattr(args, "attn_comm_ally_slots", 0)))
        self.H = max(0, int(getattr(args, "attn_comm_human_slots", 0)))
        self.register_buffer("_attn_comm_p", torch.tensor(self.P, dtype=torch.int64))
        self.register_buffer("_attn_comm_h", torch.tensor(self.H, dtype=torch.int64))
        self.msg_dim = max(8, int(getattr(args, "attn_comm_message_dim", 16)))
        self.state_dim = int(getattr(args, "attn_comm_state_dim", None) or self.hidden_size)
        nh = _n_heads_for(self.d, int(getattr(args, "attn_comm_gat_heads", 2)))
        nh_out = _n_heads_for(self.d, min(nh, 4))

        self.embed_self = nn.Linear(self.SELF_DIM, self.d)
        self.embed_ally = nn.Linear(self.ALLY_DIM, self.d)
        self.embed_recv = nn.Linear(self.msg_dim, self.d)
        self.embed_human = nn.Linear(self.HUMAN_DIM, self.d)
        self.embed_obst = nn.Linear(self.OBST_DIM, self.d)

        self.slot_fuse = nn.Linear(self.d * 2, self.d)

        self.attn_ally = nn.MultiheadAttention(self.d, nh, batch_first=True)
        self.norm_ally = nn.LayerNorm(self.d)
        self.attn_human = nn.MultiheadAttention(self.d, nh, batch_first=True)
        self.norm_human = nn.LayerNorm(self.d)

        gru_in = self.d * 3
        self.gru_cell = nn.GRUCell(gru_in, self.state_dim)

        self.out_ego_tok = nn.Linear(self.d, self.d)
        self.out_slot_tok = nn.Linear(self.d, self.d)
        self.out_q = nn.Linear(self.state_dim, self.d)
        self.out_attn = nn.MultiheadAttention(self.d, nh_out, batch_first=True)
        self.norm_out = nn.LayerNorm(self.d)

        self.msg_head = nn.Sequential(
            nn.Linear(self.state_dim + self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.msg_dim),
        )

        fuse_in = 5 * self.d + self.state_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(
        self,
        robot_obs: torch.Tensor,
        comm_rnn: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: (actor_features, comm_rnn_out, broadcast_msg) with broadcast_msg (B, msg_dim).
        """
        B = robot_obs.shape[0]
        P, Hn, M = self.P, self.H, self.msg_dim
        d = self.d
        s = 0
        self_vec = robot_obs[:, s : s + self.SELF_DIM]
        s += self.SELF_DIM
        ally = robot_obs[:, s : s + P * self.ALLY_DIM].reshape(B, P, self.ALLY_DIM)
        s += P * self.ALLY_DIM
        recv = robot_obs[:, s : s + P * M].reshape(B, P, M)
        s += P * M
        hum = robot_obs[:, s : s + Hn * self.HUMAN_DIM].reshape(B, Hn, self.HUMAN_DIM)
        s += Hn * self.HUMAN_DIM
        obst = robot_obs[:, s : s + self.OBST_DIM].reshape(B, 1, self.OBST_DIM)

        h_self = F.relu(self.embed_self(self_vec))

        if P > 0:
            ea = self.embed_ally(ally)
            er = self.embed_recv(recv)
            slot_h = F.relu(self.slot_fuse(torch.cat([ea, er], dim=-1)))
            aa, _ = self.attn_ally(slot_h, slot_h, slot_h)
            slot_h = self.norm_ally(slot_h + aa)
            ally_pool = slot_h.mean(dim=1)
        else:
            slot_h = robot_obs.new_zeros(B, 0, d)
            ally_pool = robot_obs.new_zeros(B, d)

        if Hn > 0:
            eh = self.embed_human(hum)
            ha, _ = self.attn_human(eh, eh, eh)
            ha = self.norm_human(eh + ha)
            human_pool = ha.mean(dim=1)
        else:
            human_pool = robot_obs.new_zeros(B, d)

        eo = self.embed_obst(obst)
        obst_pool = eo.squeeze(1)

        gru_in = torch.cat((h_self, ally_pool, obst_pool), dim=-1)
        h_prev = comm_rnn
        if masks is not None:
            m = masks.float().view(B, 1)
            h_prev = h_prev * m
        h_out = self.gru_cell(gru_in, h_prev)

        if P > 0:
            ego_tok = self.out_ego_tok(h_self).unsqueeze(1)
            slot_tok = self.out_slot_tok(slot_h)
            sources = torch.cat((ego_tok, slot_tok), dim=1)
            q = self.out_q(h_out).unsqueeze(1)
            out_ctx, _ = self.out_attn(q, sources, sources)
            out_ctx = self.norm_out(q + out_ctx).squeeze(1)
        else:
            out_ctx = self.out_q(h_out)

        broadcast_msg = self.msg_head(torch.cat((h_out, out_ctx), dim=-1))

        fused_in = torch.cat((h_self, ally_pool, human_pool, obst_pool, h_out, out_ctx), dim=-1)
        return self.fuse(fused_in), h_out, broadcast_msg
