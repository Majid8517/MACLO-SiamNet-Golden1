from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SparseClinicalEvidenceAttention(nn.Module):
    """
    Sparse Clinical-Conditioned Evidence Attention (SCEA).

    A clinical-conditioned query scores a compact spatial token grid derived
    from the CCRF-modulated feature map. Only the top-k token scores are kept.
    The module then adds a residual evidence vector to the CCRF representation.

    This is intentionally called "evidence" attention rather than
    lesion-supervised attention because the current classification dataset
    does not provide lesion masks for this task.
    """

    def __init__(
        self,
        channels: int = 256,
        clinical_dim: int = 256,
        pool_size: int = 6,
        keep_ratio: float = 0.25,
        dropout: float = 0.15,
    ):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")

        self.pool_size = int(pool_size)
        self.keep_ratio = float(keep_ratio)

        self.query = nn.Sequential(
            nn.Linear(clinical_dim, channels),
            nn.GELU(),
            nn.LayerNorm(channels),
        )
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)

        self.refine = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(channels),
        )

    def forward(
        self,
        feature_map: Tensor,
        clinical_token: Tensor,
        base_representation: Tensor,
    ):
        B, C, _, _ = feature_map.shape
        pooled = F.adaptive_avg_pool2d(
            feature_map, (self.pool_size, self.pool_size)
        )
        tokens = pooled.flatten(2).transpose(1, 2)  # [B,N,C]

        q = self.query(clinical_token).unsqueeze(1)  # [B,1,C]
        k = self.key(tokens)
        v = self.value(tokens)

        scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) * (C ** -0.5)
        topk = max(1, int(scores.shape[-1] * self.keep_ratio))
        top_values, top_indices = scores.topk(topk, dim=-1)

        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(1, top_indices, top_values)
        attention = torch.softmax(sparse_scores, dim=-1)

        evidence = torch.bmm(attention.unsqueeze(1), v).squeeze(1)
        evidence = self.refine(evidence)

        representation = base_representation + evidence

        # Useful audit statistics. Attention entropy is normalized to [0,1].
        eps = 1e-8
        entropy = -(attention.clamp_min(eps) * attention.clamp_min(eps).log()).sum(dim=1)
        entropy = entropy / torch.log(
            torch.tensor(float(topk), device=attention.device).clamp_min(2.0)
        )

        stats = {
            "attention_entropy": entropy,
            "topk_fraction": torch.full(
                (B,),
                float(topk) / float(scores.shape[-1]),
                dtype=feature_map.dtype,
                device=feature_map.device,
            ),
        }
        return representation, attention, stats
