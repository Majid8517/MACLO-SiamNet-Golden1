from __future__ import annotations
from typing import Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .blocks import ImageEncoderV3
from .clinical import ClinicalTokenEncoder
from .fusion import SCCTV3, AdaptiveReliabilityGate
from .ccrf import ClinicalConditionedResidualFusion
from .sparse_attention import SparseClinicalEvidenceAttention


class MACLOClassifierV3(nn.Module):
    """
    Controlled Phase-2 classifier.

    fusion_mode:
      - image_only: imaging representation only
      - concat: image + clinical simple concatenation
      - scct: SCCT-v3 without adaptive reliability gate
      - scct_gate: SCCT-v3 + reliability gate
      - ccrf: clinical-conditioned residual modulation with image-primary fusion
      - ccrf_sparse: CCRF + sparse clinical-conditioned spatial evidence attention
    """

    def __init__(
        self,
        clinical_dim: int = 29,
        num_classes: int = 2,
        channels=(32, 64, 128, 256),
        depths=(2, 2, 3, 3),
        fusion_mode: Literal[
            "image_only", "concat", "scct", "scct_gate", "ccrf", "ccrf_sparse"
        ] = "ccrf",
        dropout: float = 0.35,
        max_drop_path: float = 0.15,
        ccrf_strength: float = 0.35,
        sparse_pool_size: int = 6,
        sparse_keep_ratio: float = 0.25,
        clinical_encoder_type: Literal["deep", "linear"] = "deep",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.clinical_encoder_type = clinical_encoder_type
        dim = channels[-1]

        self.image_encoder = ImageEncoderV3(
            1, channels, depths, max_drop_path=max_drop_path
        )
        self.image_norm = nn.LayerNorm(dim)

        self.clinical_encoder = None
        if fusion_mode != "image_only":
            if clinical_encoder_type == "deep":
                self.clinical_encoder = ClinicalTokenEncoder(
                    input_dim=clinical_dim,
                    token_dim=dim,
                    hidden_dim=128,
                    dropout=0.25,
                )
            elif clinical_encoder_type == "linear":
                # Controlled ablation: a single linear projection with no hidden MLP.
                self.clinical_encoder = nn.Linear(clinical_dim, dim)
            else:
                raise ValueError(
                    f"Unsupported clinical_encoder_type={clinical_encoder_type}"
                )

        self.concat_proj = None
        self.scct = None
        self.reliability_gate = None
        self.ccrf = None
        self.sparse_attention = None

        if fusion_mode == "concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim),
            )

        if fusion_mode in {"scct", "scct_gate"}:
            self.scct = SCCTV3(dim=dim, heads=4, depth=2, dropout=0.15)

        if fusion_mode == "scct_gate":
            self.reliability_gate = AdaptiveReliabilityGate(dim=dim)

        if fusion_mode in {"ccrf", "ccrf_sparse"}:
            self.ccrf = ClinicalConditionedResidualFusion(
                channels=dim,
                clinical_dim=dim,
                hidden=128,
                modulation_strength=ccrf_strength,
                dropout=0.20,
            )

        if fusion_mode == "ccrf_sparse":
            self.sparse_attention = SparseClinicalEvidenceAttention(
                channels=dim,
                clinical_dim=dim,
                pool_size=sparse_pool_size,
                keep_ratio=sparse_keep_ratio,
                dropout=0.15,
            )

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
        modulation_stats = None
        sparse_attention = None
        sparse_stats = None
        fused_feature_map = top

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

            elif self.fusion_mode in {"ccrf", "ccrf_sparse"}:
                representation, fused_feature_map, modulation_stats = self.ccrf(
                    top, image_global, clinical_token
                )

                if self.fusion_mode == "ccrf_sparse":
                    representation, sparse_attention, sparse_stats = self.sparse_attention(
                        fused_feature_map,
                        clinical_token,
                        representation,
                    )

            else:
                raise ValueError(f"Unsupported fusion_mode={self.fusion_mode}")

        logits = self.head(representation)
        return {
            "logits": logits,
            "embedding": representation,
            "gate_weights": gate_weights,
            "modulation_stats": modulation_stats,
            "sparse_attention": sparse_attention,
            "sparse_stats": sparse_stats,
            "top_feature": top,
            "fused_feature_map": fused_feature_map,
        }
