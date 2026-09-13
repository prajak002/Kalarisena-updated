"""Successor-Conditioned Viability Critic network (paper Sec 3.3).

3-layer MLP (512, 256, 128), matching the paper's stated architecture.
Trained with BCE(V_psi, Y_via) + lambda_B * Brier(V_psi, Y_via) as specified.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViabilityCritic(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def viability_loss(v_pred: torch.Tensor, y_true: torch.Tensor, lambda_brier: float = 1.0) -> torch.Tensor:
    eps = 1e-7
    v = v_pred.clamp(eps, 1 - eps)
    bce = -(y_true * torch.log(v) + (1 - y_true) * torch.log(1 - v)).mean()
    brier = ((v_pred - y_true) ** 2).mean()
    return bce + lambda_brier * brier
