from __future__ import annotations
from typing import Dict, List, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SelectiveCrossContextTransformer(nn.Module):
    """
    SCCT v2: compact, genuine Transformer fusion over modality/context tokens.

    Spatial feature maps are first summarized into one token per modality.
    The Transformer therefore scales with the number of modalities rather than H*W,
    avoiding the memory explosion of full 256x256 self-attention.
    """
    def __init__(
        self,
        channels: int,
        modality_names: List[str],
        meta_dim: int = 0,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.modality_names = list(modality_names)
        self.modality_embedding = nn.Parameter(
            torch.randn(len(self.modality_names), channels) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

        self.meta_proj = nn.Linear(meta_dim, channels) if meta_dim > 0 else None
        self.meta_embedding = (
            nn.Parameter(torch.randn(1, 1, channels) * 0.02)
            if meta_dim > 0 else None
        )
        self.context_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.out_norm = nn.GroupNorm(min(8, channels), channels)

    def forward(
        self,
        features: Dict[str, Tensor],
        availability: Tensor,
        metadata: Optional[Tensor] = None,
    ):
        if availability.dtype != torch.bool:
            availability = availability > 0

        if availability.ndim != 2 or availability.shape[1] != len(self.modality_names):
            raise ValueError("availability must have shape [B, number_of_modalities].")
        if not torch.all(availability.any(dim=1)):
            raise ValueError("Every sample must contain at least one available imaging modality.")

        ref = next(iter(features.values()))
        B, C, H, W = ref.shape

        modality_tokens, modality_maps = [], []
        for idx, name in enumerate(self.modality_names):
            fmap = features[name]
            modality_maps.append(fmap)
            token = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            token = token + self.modality_embedding[idx].unsqueeze(0)
            modality_tokens.append(token)

        tokens = torch.stack(modality_tokens, dim=1)
        valid = availability

        if self.meta_proj is not None and metadata is not None:
            meta_token = self.meta_proj(metadata).unsqueeze(1) + self.meta_embedding
            tokens = torch.cat([tokens, meta_token], dim=1)
            valid = torch.cat(
                [valid, torch.ones(B, 1, dtype=torch.bool, device=valid.device)],
                dim=1,
            )

        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        weights = valid.float().unsqueeze(-1)
        context = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)

        spatial_stack = torch.stack(modality_maps, dim=1)
        spatial_weights = availability.float().view(B, len(self.modality_names), 1, 1, 1)
        spatial = (
            (spatial_stack * spatial_weights).sum(1)
            / spatial_weights.sum(1).clamp_min(1.0)
        )

        gate = self.context_gate(context).view(B, C, 1, 1)
        fused = self.out_norm(spatial * (1.0 + gate))
        return fused, context
