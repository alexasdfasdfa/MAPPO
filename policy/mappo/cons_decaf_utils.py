"""
Cons-DecAF (Xiang et al., arXiv:2307.12287) training helpers: global soft labels e_g and KL loss L_CE.
Policy distillation MSE for (logits, h) vs teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_ce_loss_paper(e_g: torch.Tensor, logits_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L_CE = E[ sum_x e_g(x) log(e_g(x) / hat_e(x)) ] with hat_e = softmax(logits_hat).
    Same as KL(e_g || softmax(logits_hat)) for fixed e_g (detached target).
    """
    log_hat = F.log_softmax(logits_hat, dim=-1)
    log_e = (e_g + eps).log()
    return (e_g * (log_e - log_hat)).sum(dim=-1).mean()


class FrozenGlobalLabelProjector(nn.Module):
    """
    Maps flattened centralized observation to K logits; frozen (not trainable).
    e_g = softmax(proj(share_obs)) is the supervised target for the CE global estimator.
    """

    def __init__(self, in_dim: int, k_bins: int):
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(k_bins), bias=True)
        nn.init.orthogonal_(self.proj.weight, gain=0.01)
        nn.init.zeros_(self.proj.bias)
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, share_obs_flat: torch.Tensor) -> torch.Tensor:
        z = self.proj(share_obs_flat)
        return F.softmax(z, dim=-1)


def mse_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
) -> torch.Tensor:
    """Eq.(17): mean squared error on stacked action logits and consensus embedding h."""
    return F.mse_loss(student_logits, teacher_logits) + F.mse_loss(student_h, teacher_h)
