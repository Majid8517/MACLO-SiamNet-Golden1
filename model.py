"""
model.py

Compact reference implementation of MACLO-SiamNet used in the
reproducibility code:

- MRI branch  : TwoStageFocalTransformerHopfield
- CT branch   : ModifiedEfficientViT
- Fusion      : SCCTFusion (MRI + CT [+ optional metadata])
- Heads       : segmentation, classification, age regression

Forward interface is designed to be compatible with the evaluation scripts:

    seg, cls, age = model(mri, ct)                 # inference (no metadata)
    Z, seg, cls, age = model(mri, ct, meta, True)  # training with MACLO

where:
    mri:  [B, 1, H, W]
    ct:   [B, 1, H, W]
    meta: [B, meta_dim] or None
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .backbones import (
    TwoStageFocalTransformerHopfield,
    ModifiedEfficientViT,
    SCCTFusion,
)


class SegHead(nn.Module):
    """
    Simple segmentation head on top of fused feature map.

    Input:  F_fused ∈ R^{B × C × H × W}
    Output: seg_logits ∈ R^{B × 1 × H × W}
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ClsHead(nn.Module):
    """
    Classification head on top of global pooled shared representation Z.

    Input:  Z ∈ R^{B × C}
    Output: logits ∈ R^{B × num_classes}
    """

    def __init__(self, in_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: Tensor) -> Tensor:
        return self.fc(z)


class AgeHead(nn.Module):
    """
    Age regression head on top of global pooled shared representation Z.

    Input:  Z ∈ R^{B × C}
    Output: age_pred ∈ R^{B × 1}
    """

    def __init__(self, in_dim: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: Tensor) -> Tensor:
        return self.fc(z)


class MACLOSiamNet(nn.Module):
    """
    MACLO-SiamNet (compact backbone version):

    MRI  -> TwoStageFocalTransformerHopfield  -> F_mri ∈ R^{B×C×H×W}
    CT   -> ModifiedEfficientViT             -> F_ct  ∈ R^{B×C×H×W}
    Fuse -> SCCTFusion                       -> F_fused ∈ R^{B×C×H×W}
    GAP(F_fused) -> Z_shared ∈ R^{B×C}

    Heads:
        - SegHead(F_fused)      -> seg_logits ∈ R^{B×1×H×W}
        - ClsHead(Z_shared)     -> cls_logits ∈ R^{B×num_classes}
        - AgeHead(Z_shared)     -> age_pred   ∈ R^{B×1}

    Forward API:
        seg, cls, age = model(mri, ct)
        Z, seg, cls, age = model(mri, ct, meta, return_z=True)
    """

    def __init__(
        self,
        img_size: int = 256,
        base_channels: int = 32,
        num_classes: int = 2,
        meta_dim: int = 0,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.c = base_channels
        self.num_classes = num_classes
        self.meta_dim = meta_dim

        # MRI / CT encoders
        self.mri_backbone = TwoStageFocalTransformerHopfield(
            c_in=1,
            c=base_channels,
        )
        self.ct_backbone = ModifiedEfficientViT(
            c_in=1,
            c=base_channels,
        )

        # Fusion (MRI + CT [+ optional metadata])
        self.fusion = SCCTFusion(
            c=base_channels,
            meta_dim=meta_dim,
        )

        # Heads
        self.seg_head = SegHead(in_channels=base_channels, out_channels=1)
        self.cls_head = ClsHead(in_dim=base_channels, num_classes=num_classes)
        self.age_head = AgeHead(in_dim=base_channels)

    def _global_pool(self, x: Tensor) -> Tensor:
        """
        Global average pooling over H,W:
            x: [B, C, H, W] -> [B, C]
        """
        return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)

    def forward(
        self,
        mri: Tensor,
        ct: Tensor,
        meta: Optional[Tensor] = None,
        return_z: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            mri:   MRI images, shape [B, 1, H, W]
            ct:    CT images,  shape [B, 1, H, W]
            meta:  optional metadata tensor [B, meta_dim] (if used in fusion)
            return_z:
                If True, also return Z_shared for MACLO-style gradient weighting.

        Returns:
            If return_z == False:
                seg_logits, cls_logits, age_pred
            If return_z == True:
                Z_shared, seg_logits, cls_logits, age_pred
        """

        # --- Encoder branches ---
        fmri = self.mri_backbone(mri)  # [B, C, H, W]
        fct = self.ct_backbone(ct)     # [B, C, H, W]

        # --- Fusion ---
        if self.meta_dim > 0 and meta is not None:
            z_fused = self.fusion(fmri, fct, metadata=meta)  # [B, C, H, W]
        else:
            z_fused = self.fusion(fmri, fct, metadata=None)  # [B, C, H, W]

        # --- Shared representation ---
        Z_shared = self._global_pool(z_fused)                # [B, C]

        # --- Heads ---
        seg_logits = self.seg_head(z_fused)                  # [B, 1, H, W]
        cls_logits = self.cls_head(Z_shared)                 # [B, num_classes]
        age_pred = self.age_head(Z_shared)                   # [B, 1]

        if return_z:
            return Z_shared, seg_logits, cls_logits, age_pred
        else:
            # for evaluation scripts: seg, cls, age = model(mri, ct)
            return seg_logits, cls_logits, age_pred
