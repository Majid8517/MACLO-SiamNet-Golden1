from __future__ import annotations
import torch
from torch import nn, Tensor


class ClinicalTokenEncoder(nn.Module):
    """
    Encodes the leakage-audited 29-D clinical vector into a contextual token.

    LayerNorm is used instead of BatchNorm so behavior remains stable with
    small medical-imaging batch sizes.
    """
    def __init__(
        self,
        input_dim: int = 29,
        token_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
