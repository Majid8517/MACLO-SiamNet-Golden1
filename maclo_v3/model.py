from __future__ import annotations
from typing import Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .blocks import ImageEncoderV3
from .clinical import ClinicalTokenEncoder
from .fusion import SCCTV3, AdaptiveReliabilityGate


class MACLOClassifierV3(nn.Module):
    """
    Controlled Phase-2 classifier.

    fusion_mode:
      - image_only: imaging representation only
      - concat: image + clinical simple concatenation
      - scct: SCCT-v3 without adaptive reliability gate
      - scct_gate: full Phase-2 fusion module

    These modes permit controlled ablations with the same image encoder,
    clinical encoder and classifier capacity family.
    """
    def __init__(
        self,
        clinical_dim: int = 29,
        num_classes: int = 2,
        channels=(32, 64, 128, 256),
        depths=(2, 2, 3, 3),
        fusion_mode: Literal["image_only", "concat", "scct", "scct_gate"] = "scct_gate",
        dropout: float = 0.35,
        max_drop_path: float = 0.15,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        dim = channels[-1]

        self.image_encoder = ImageEncoderV3(
            1, channels, depths, max_drop_path=max_drop_path
        )
        self.image_norm = nn.LayerNorm(dim)

        self.clinical_encoder = None
        if fusion_mode != "image_only":
            self.clinical_encoder = ClinicalTokenEncoder(
                input_dim=clinical_dim,
                token_dim=dim,
                hidden_dim=128,
                dropout=0.25,
            )

        if fusion_mode == "concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim),
            )
        else:
            self.concat_proj = None

        if fusion_mode in {"scct", "scct_gate"}:
            self.scct = SCCTV3(dim=dim, heads=4, depth=2, dropout=0.15)
        else:
            self.scct = None

        if fusion_mode == "scct_gate":
            self.reliability_gate = AdaptiveReliabilityGate(dim=dim)
        else:
            self.reliability_gate = None

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: Tensor, clinical: Tensor | None = None):
        top = self.image_encoder(image)[-1]
        image_global = F.adaptive_avg_pool2d(top, 1).flatten(1)
        image_global = self.image_norm(image_global)

        gate_weights = None

        if self.fusion_mode == "image_only":
            representation = image_global

        else:
            if clinical is None:
                raise ValueError("clinical input is required for this fusion mode.")
            clinical_token = self.clinical_encoder(clinical)

            if self.fusion_mode == "concat":
                representation = self.concat_proj(
                    torch.cat([image_global, clinical_token], dim=1)
                )

            elif self.fusion_mode == "scct":
                representation, _, _ = self.scct(top, clinical_token)

            elif self.fusion_mode == "scct_gate":
                fused, image_ctx, clinical_ctx = self.scct(top, clinical_token)
                representation, gate_weights = self.reliability_gate(
                    fused, image_ctx, clinical_ctx
                )

            else:
                raise ValueError(f"Unsupported fusion_mode={self.fusion_mode}")

        logits = self.head(representation)
        return {
            "logits": logits,
            "embedding": representation,
            "gate_weights": gate_weights,
            "top_feature": top,
        }
