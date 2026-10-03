from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SCCTV3(nn.Module):
    """
    Cross-context transformer for image-clinical fusion.

    A compact 4x4 token grid is extracted from the top image feature map.
    A clinical token and a learnable fusion token are concatenated with those
    image tokens. Transformer attention is therefore O((16+2)^2), not O((HW)^2).
    """
    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.15,
        token_grid: int = 4,
    ):
        super().__init__()
        self.token_grid = token_grid
        self.fusion_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.clinical_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.image_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        max_tokens = token_grid * token_grid + 2
        self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, image_map: Tensor, clinical_token: Tensor):
        B, C, _, _ = image_map.shape
        pooled = F.adaptive_avg_pool2d(
            image_map, (self.token_grid, self.token_grid)
        )
        image_tokens = pooled.flatten(2).transpose(1, 2) + self.image_type
        clinical = clinical_token.unsqueeze(1) + self.clinical_type
        fusion = self.fusion_token.expand(B, -1, -1)

        tokens = torch.cat([fusion, clinical, image_tokens], dim=1)
        tokens = tokens + self.pos[:, :tokens.shape[1]]
        encoded = self.encoder(tokens)
        fused = self.out_norm(encoded[:, 0])
        clinical_ctx = self.out_norm(encoded[:, 1])
        image_ctx = self.out_norm(encoded[:, 2:].mean(dim=1))
        return fused, image_ctx, clinical_ctx


class AdaptiveReliabilityGate(nn.Module):
    """
    Learns sample-specific image-vs-clinical reliability without assuming that
    one modality is always dominant.
    """
    def __init__(self, dim: int = 256, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        fused_context: Tensor,
        image_context: Tensor,
        clinical_context: Tensor,
    ):
        logits = self.gate(
            torch.cat([fused_context, image_context, clinical_context], dim=1)
        )
        weights = torch.softmax(logits, dim=1)
        mixed = (
            weights[:, 0:1] * image_context
            + weights[:, 1:2] * clinical_context
        )
        return self.refine(mixed + fused_context), weights
