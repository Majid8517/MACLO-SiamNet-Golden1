from __future__ import annotations
from typing import Dict, List, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .blocks import ModalityStem, SharedEncoder, ConvNormAct
from .scct import SelectiveCrossContextTransformer
from .spam import SparsePeerAttention


class SegDecoder(nn.Module):
    def __init__(self, c4=256, c3=128, c2=64, c1=32):
        super().__init__()
        self.up3 = ConvNormAct(c4 + c3, c3)
        self.up2 = ConvNormAct(c3 + c2, c2)
        self.up1 = ConvNormAct(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, f4: Tensor, f3: Tensor, f2: Tensor, f1: Tensor):
        x = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)


class MACLOSiamNetV2(nn.Module):
    """
    Modality-adaptive redesign.

    Default modalities target CT-domain experiments:
      NCCT + optional CTP maps (CBF, CBV, MTT, Tmax).

    Classification is optional and disabled by default until a defensible
    label definition is established from the source datasets.
    """
    def __init__(
        self,
        modality_names: List[str] = ("ncct", "cbf", "cbv", "mtt", "tmax"),
        channels=(32, 64, 128, 256),
        meta_dim: int = 0,
        num_classes: Optional[int] = None,
        spam_keep_ratio: float = 0.25,
        scct_heads: int = 4,
        scct_depth: int = 2,
    ):
        super().__init__()
        self.modality_names = list(modality_names)
        self.channels = tuple(channels)

        self.stems = nn.ModuleDict({
            name: ModalityStem(self.channels[0]) for name in self.modality_names
        })
        self.shared_encoder = SharedEncoder(self.channels)

        self.scct = SelectiveCrossContextTransformer(
            channels=self.channels[-1],
            modality_names=self.modality_names,
            meta_dim=meta_dim,
            num_heads=scct_heads,
            depth=scct_depth,
        )

        tasks = ["seg", "age"] + (["cls"] if num_classes is not None else [])
        self.spam = SparsePeerAttention(
            channels=self.channels[-1],
            tasks=tasks,
            keep_ratio=spam_keep_ratio,
        )

        self.seg_context_gate = nn.Linear(self.channels[-1], self.channels[-1])
        self.seg_decoder = SegDecoder(
            self.channels[3], self.channels[2], self.channels[1], self.channels[0]
        )

        self.age_head = nn.Sequential(
            nn.Linear(self.channels[-1] * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.cls_head = None
        if num_classes is not None:
            self.cls_head = nn.Sequential(
                nn.Linear(self.channels[-1] * 2, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

    def shared_parameters(self):
        """Parameters coordinated by MACLO."""
        modules = [self.stems, self.shared_encoder, self.scct]
        for module in modules:
            yield from module.parameters()

    def _masked_scale_average(
        self,
        all_features: Dict[str, List[Tensor]],
        scale_index: int,
        availability: Tensor,
    ) -> Tensor:
        maps = [all_features[m][scale_index] for m in self.modality_names]
        stack = torch.stack(maps, dim=1)
        w = availability.float().view(
            availability.shape[0], len(self.modality_names), 1, 1, 1
        )
        return (stack * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(
        self,
        inputs: Dict[str, Tensor],
        availability: Tensor,
        metadata: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if set(inputs.keys()) != set(self.modality_names):
            raise ValueError(
                "inputs must contain every configured modality key; use zero tensors "
                "for unavailable modalities and mark them false in availability."
            )

        all_features = {}
        for name in self.modality_names:
            all_features[name] = self.shared_encoder(self.stems[name](inputs[name]))

        f1 = self._masked_scale_average(all_features, 0, availability)
        f2 = self._masked_scale_average(all_features, 1, availability)
        f3 = self._masked_scale_average(all_features, 2, availability)

        high = {name: all_features[name][-1] for name in self.modality_names}
        fused, shared_context = self.scct(high, availability, metadata)
        task_context = self.spam(fused)

        seg_gate = torch.sigmoid(
            self.seg_context_gate(task_context["seg"])
        ).unsqueeze(-1).unsqueeze(-1)
        seg_map = fused * (1.0 + seg_gate)
        seg_logits = self.seg_decoder(seg_map, f3, f2, f1)

        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        age_pred = self.age_head(
            torch.cat([pooled, task_context["age"]], dim=1)
        ).squeeze(-1)

        out = {
            "seg": seg_logits,
            "age": age_pred,
            "shared_map": fused,
            "shared_context": shared_context,
        }

        if self.cls_head is not None:
            out["cls"] = self.cls_head(
                torch.cat([pooled, task_context["cls"]], dim=1)
            )

        return out
