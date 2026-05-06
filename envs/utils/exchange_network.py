"""
ExchangeNetwork: predicts swap decisions for agent pairs, trained via PPO.
"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from envs.utils.utils import cal_distance


# ---------------------------------------------------------------------------
# ExchangeNetwork
# ---------------------------------------------------------------------------

class ExchangeNetwork(nn.Module):
    """
    Predicts whether two agents should swap their discrete goal targets.

    Input features (per pair, concatenated = 20 dims):
        agent_i: gx-px, gy-py, v, theta, for_feature, vx_for, vy_for, px, py  (9)
        agent_j: same layout                                                   (9)
        derived: dist_ij, mutual_improve_score                                 (2)

    Output: single logit (sigmoid -> swap probability).
    """

    AGENT_FEAT_DIM = 9  # 7 base + 2 position
    PAIR_FEAT_DIM = AGENT_FEAT_DIM * 2 + 2  # 20

    def __init__(self, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.PAIR_FEAT_DIM, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """pair_features: [batch, 20] -> [batch, 1] logit."""
        return self.net(pair_features)

    @staticmethod
    def build_pair_features(
        obs_i: np.ndarray,
        obs_j: np.ndarray,
        px_i: float, py_i: float,
        px_j: float, py_j: float,
        gx_i: float, gy_i: float,
        gx_j: float, gy_j: float,
    ) -> np.ndarray:
        """
        Build the 20-dim feature vector for one (i, j) pair.

        Parameters:
            obs_i, obs_j: full observation rows (first 9 dims used)
            px/py: world positions
            gx/gy: goal positions
        Returns:
            np.ndarray of shape (20,)
        """
        feat_i = np.array(obs_i[:9], dtype=np.float32)
        feat_j = np.array(obs_j[:9], dtype=np.float32)

        dist_ij = float(cal_distance(px_i, py_i, px_j, py_j))

        # mutual_improve: positive means both agents get closer after swap
        d_i_current = cal_distance(px_i, py_i, gx_i, gy_i)
        d_j_current = cal_distance(px_j, py_j, gx_j, gy_j)
        d_i_after = cal_distance(px_i, py_i, gx_j, gy_j)
        d_j_after = cal_distance(px_j, py_j, gx_i, gy_i)
        mutual_improve = float(
            (d_i_current - d_i_after) + (d_j_current - d_j_after)
        )

        return np.concatenate([feat_i, feat_j, [dist_ij, mutual_improve]]).astype(np.float32)

    @torch.no_grad()
    def predict_swap_prob(self, pair_features: np.ndarray) -> np.ndarray:
        """
        pair_features: [batch, 20] numpy array
        Returns: [batch] swap probabilities in [0, 1]
        """
        x = torch.from_numpy(pair_features)
        logit = self.forward(x)
        prob = torch.sigmoid(logit).squeeze(-1).numpy()
        return prob

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        self.to(device)
        self.eval()

    def get_swap_log_prob(self, pair_features: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of swap action for pair features.

        Args:
            pair_features: [batch, 20] tensor
        Returns:
            log_prob: [batch, 2] tensor, columns are [log_prob(no_swap), log_prob(swap)]
        """
        logit = self.forward(pair_features)  # [batch, 1]
        prob = torch.sigmoid(logit)
        # Bernoulli: log_prob = [log(1-p), log(p)]
        log_prob = torch.cat([torch.log(1 - prob + 1e-8), torch.log(prob + 1e-8)], dim=-1)
        return log_prob

    def sample_swap(self, pair_features: torch.Tensor) -> tuple:
        """
        Sample swap action and return (action, log_prob).

        Args:
            pair_features: [batch, 20] tensor
        Returns:
            action: [batch] tensor of 0/1
            log_prob: [batch] tensor of log probability of sampled action
        """
        logit = self.forward(pair_features)  # [batch, 1]
        prob = torch.sigmoid(logit).squeeze(-1)  # [batch]
        dist = torch.distributions.Bernoulli(prob)
        action = dist.sample()  # [batch]
        log_prob = dist.log_prob(action)  # [batch]
        return action, log_prob

    def compute_entropy(self, pair_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Bernoulli entropy of swap distribution.

        If pair_features is provided, computes entropy for those features.
        Otherwise computes from current output (requires forward pass context).

        Returns:
            entropy: [batch] tensor of Bernoulli entropy values
        """
        if pair_features is None:
            # This method is typically called with features during loss computation
            raise ValueError("pair_features must be provided")
        logit = self.forward(pair_features)
        prob = torch.sigmoid(logit)
        # Bernoulli entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
        return entropy.squeeze(-1)
