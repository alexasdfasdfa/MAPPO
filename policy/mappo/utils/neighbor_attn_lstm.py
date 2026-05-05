import torch
import torch.nn as nn


class NeighborSelfAttnLSTM(nn.Module):
    """
    Encode nearest-teammate rows: linear embed -> self-attention over the set ->
    LSTM along the distance-ordered sequence -> last-layer hidden -> linear.
    Intended for dynamic-goal actor branch (coordination / target switching).
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        hidden_size: int,
        n_heads: int,
        lstm_layers: int,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}")
        self.hidden_size = hidden_size
        self.embed = nn.Linear(feat_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=max(1, int(lstm_layers)),
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, neighbor_feats: torch.Tensor) -> torch.Tensor:
        # neighbor_feats: (B, P, F); P may be 0
        bsz = neighbor_feats.shape[0]
        if neighbor_feats.shape[1] == 0:
            return neighbor_feats.new_zeros(bsz, self.hidden_size)
        h = self.embed(neighbor_feats)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)
        _, (hn, _) = self.lstm(h)
        return self.out(hn[-1])
