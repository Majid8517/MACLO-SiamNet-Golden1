from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ClinicalConditionedResidualFusion(nn.Module):
    """
    CCRF: Clinical-Conditioned Residual Fusion.

    The image pathway remains primary. Clinical metadata modulates image
    features through bounded FiLM-style scale/shift terms and contributes
    a residual clinical context. This prevents the fusion block from simply
    replacing image evidence with tabular metadata.

    F_mod = F_img * (1 + lambda * tanh(gamma)) + lambda * tanh(beta)
    z_fused = LN(z_img + Wc(z_clin) + Wr(GAP(F_mod)))
    """

    def __init__(
        self,
        channels: int = 256,
        clinical_dim: int = 256,
        hidden: int = 128,
        modulation_strength: float = 0.35,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.modulation_strength = float(modulation_strength)

        self.conditioner = nn.Sequential(
            nn.Linear(clinical_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels * 2),
        )

        self.clinical_residual = nn.Sequential(
            nn.Linear(clinical_dim, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.image_residual = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out_norm = nn.LayerNorm(channels)

    def forward(
        self,
        image_map: Tensor,
        image_global: Tensor,
        clinical_token: Tensor,
    ):
        B, C, _, _ = image_map.shape

        gamma_beta = self.conditioner(clinical_token)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(B, C, 1, 1)
        beta = torch.tanh(beta).view(B, C, 1, 1)

        lam = self.modulation_strength
        modulated = image_map * (1.0 + lam * gamma) + lam * beta
        modulated_global = F.adaptive_avg_pool2d(modulated, 1).flatten(1)

        representation = self.out_norm(
            image_global
            + self.clinical_residual(clinical_token)
            + self.image_residual(modulated_global)
        )

        modulation_stats = {
            "gamma_abs_mean": gamma.abs().mean(dim=(1, 2, 3)),
            "beta_abs_mean": beta.abs().mean(dim=(1, 2, 3)),
        }
        return representation, modulated, modulation_stats
