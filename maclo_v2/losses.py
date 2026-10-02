from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor
import torch.nn.functional as F


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.float()
    if mask.sum().item() == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / mask.sum()


def dice_bce_per_sample(logits: Tensor, target: Tensor, eps: float = 1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).mean(dims)
    return dice_loss + bce


def compute_task_losses(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    task_mask: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    losses = {}

    if "seg" in outputs:
        per_sample = dice_bce_per_sample(outputs["seg"], targets["seg"].float())
        losses["seg"] = _masked_mean(per_sample, task_mask["seg"])

    if "age" in outputs:
        per_sample = (outputs["age"] - targets["age"].float()).pow(2)
        losses["age"] = _masked_mean(per_sample, task_mask["age"])

    if "cls" in outputs and "cls" in targets:
        per_sample = F.cross_entropy(
            outputs["cls"], targets["cls"].long(), reduction="none"
        )
        losses["cls"] = _masked_mean(per_sample, task_mask["cls"])

    return losses
