backbones.py

Backbone modules for MACLO-SiamNet:

- TwoStageFocalTransformerHopfield:
    MRI encoder (NF-Transformer style) with ConvBNAct + focal-like attention
    + Hopfield-like associative refinement in two consecutive stages.

- ModifiedEfficientViT:
    Lightweight CT encoder based on depthwise‐separable convolutions and
    attention, used as a compressed version of the STGE-EfficientViT.

- SCCTFusion:
    Soft-Compact Convolutional Transformer fusion block that:
        * fuses MRI / CT feature maps
        * optionally injects clinical metadata
        * applies deformable-like mixing for geometry-aware fusion.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from .blocks import (
    ConvBNAct,
    SimpleAttention,
    HopfieldLike,
    DepthwiseSeparableConv,
    DeformableLike,
)

__all__ = [
    "TwoStageFocalTransformerHopfield",
    "ModifiedEfficientViT",
    "SCCTFusion",
]


class TwoStageFocalTransformerHopfield(nn.Module):
    """
    MRI encoder used in MACLO-SiamNet.

    This module implements a compact version of the proposed NF-Transformer:
    each stage consists of:
        ConvBNAct  -> SimpleAttention -> HopfieldLike

    The SimpleAttention block is assumed to implement a neuro-focal style
    attention, while HopfieldLike adds attractor-based refinement.

    Args:
        c_in:  Number of input channels (e.g., 1 for single-channel MRI).
        c:     Base number of feature channels.
    """

    def __init__(self, c_in: int = 1, c: int = 32) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBNAct(c_in, c),
            SimpleAttention(c),
            HopfieldLike(c),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(c, c),
            SimpleAttention(c),
            HopfieldLike(c),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, c_in, H, W]

        Returns:
            Tensor of shape [B, c, H, W] – MRI feature embedding.
        """
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class ModifiedEfficientViT(nn.Module):
    """
    CT encoder used in MACLO-SiamNet.

    This module provides a lightweight EfficientViT-style backbone based on:
        - depthwise-separable convolutions for efficient spatial encoding
        - SimpleAttention for global context modelling.

    It acts as a compressed implementation of the STGE-EfficientViT described
    in the paper.

    Args:
        c_in:  Number of input channels for CT (e.g., 1 for NCCT).
        c:     Base number of feature channels.
    """

    def __init__(self, c_in: int = 1, c: int = 32) -> None:
        super().__init__()

        # Stem: initial projection
        self.stem = DepthwiseSeparableConv(c_in, c)

        # Encoder: additional depthwise conv + attention
        self.enc = nn.Sequential(
            DepthwiseSeparableConv(c, c),
            SimpleAttention(c),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, c_in, H, W]

        Returns:
            Tensor of shape [B, c, H, W] – CT feature embedding.
        """
        x = self.stem(x)
        x = self.enc(x)
        return x


class SCCTFusion(nn.Module):
    """
    Soft-Compact Convolutional Transformer (SCCT) fusion block.

    This module fuses MRI and CT feature maps and optionally injects
    clinical metadata into the shared representation. The fusion is:

        1. Concatenate MRI & CT features along the channel dimension.
        2. 1x1 Conv to reduce channels back to c (learned fusion).
        3. DeformableLike block for geometry-aware mixing.
        4. Optional metadata projection added as a global bias.

    Args:
        c:        Number of feature channels for each modality.
        meta_dim: Dimensionality of the metadata vector (0 if unused).

    Expected shapes:
        fmri:     [B, c,  H, W]
        fct:      [B, c,  H, W]
        metadata: [B, meta_dim] or [B, meta_dim, ...] (will be flattened)
    """

    def __init__(self, c: int = 32, meta_dim: int = 0) -> None:
        super().__init__()

        # Fuse MRI + CT by concatenation then 1x1 conv
        self.reduce = nn.Conv2d(2 * c, c, kernel_size=1, bias=True)

        # Deformable-like geometry-aware mixing
        self.mix = DeformableLike(c)

        # Optional metadata projection
        self.meta_dim: int = meta_dim
        self.meta_proj: Optional[nn.Linear]
        if meta_dim > 0:
            self.meta_proj = nn.Linear(meta_dim, c)
        else:
            self.meta_proj = None

    def forward(
        self,
        fmri: Tensor,
        fct: Tensor,
        metadata: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            fmri:     MRI feature map, shape [B, c, H, W].
            fct:      CT feature map,  shape [B, c, H, W].
            metadata: Optional clinical metadata tensor.

        Returns:
            Tensor of shape [B, c, H, W] – fused multimodal representation.
        """
        # 1) Concatenate MRI and CT features along channels
        z = torch.cat([fmri, fct], dim=1)  # [B, 2c, H, W]

        # 2) Channel reduction + deformable mixing
        z = self.reduce(z)  # [B, c, H, W]
        z = self.mix(z)     # [B, c, H, W]

        # 3) Optional metadata injection
        if self.meta_proj is not None and metadata is not None:
            # Flatten metadata to [B, meta_dim] if it has extra dims
            if metadata.dim() > 2:
                metadata = metadata.view(metadata.size(0), -1)

            m = self.meta_proj(metadata)          # [B, c]
            m = m.unsqueeze(-1).unsqueeze(-1)     # [B, c, 1, 1]
            z = z + m                             # broadcast over H, W

        return z
