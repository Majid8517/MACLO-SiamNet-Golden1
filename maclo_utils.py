"""
maclo_utils.py

Utility functions for MACLO-SiamNet:

- compute_task_losses:
    Computes per-task losses for:
        * segmentation (Dice loss over sigmoid probabilities)
        * classification (CrossEntropy)
        * lesion-age regression (MAE)

- maclo_unified_loss:
    Simplified MACLO-style unified loss that:
        * derives gradient vectors for each task w.r.t. a shared feature Z_shared
        * computes cosine similarities (synergy) between task gradients
        * computes difficulty-aware weights based on gradient norms
        * combines them into unified task weights λ_seg, λ_cls, λ_age
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def compute_task_losses(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute individual task losses.

    Args:
        outputs: dict with keys ['seg', 'cls', 'age']:
            - 'seg': [B, 1, H, W] (logits)
            - 'cls': [B, num_classes]
            - 'age': [B, 1] or [B]
        targets: dict with keys ['seg', 'cls', 'age']:
            - 'seg': [B, 1, H, W] binary mask in {0,1}
            - 'cls': [B] class indices (long)
            - 'age': [B] lesion age (float)

    Returns:
        L_seg, L_cls, L_age  (scalar tensors)
    """

    # ----- segmentation: Dice loss over sigmoid probabilities -----
    seg_logits = outputs["seg"]             # [B, 1, H, W]
    seg_gt = targets["seg"].float()         # [B, 1, H, W]

    seg_prob = torch.sigmoid(seg_logits)
    smooth = 1e-5
    intersection = (seg_prob * seg_gt).sum()
    dice = (2.0 * intersection + smooth) / (
        seg_prob.sum() + seg_gt.sum() + smooth
    )
    L_seg = 1.0 - dice

    # ----- classification: CrossEntropy -----
    cls_pred = outputs["cls"]              # [B, num_classes]
    cls_gt = targets["cls"].long()         # [B]
    L_cls = F.cross_entropy(cls_pred, cls_gt)

    # ----- age regression: MAE (L1) -----
    age_pred = outputs["age"].squeeze(-1)  # [B]
    age_gt = targets["age"].float()        # [B]
    L_age = F.l1_loss(age_pred, age_gt)

    return L_seg, L_cls, L_age


def maclo_unified_loss(
    Z_shared: Tensor,
    losses: Tuple[Tensor, Tensor, Tensor],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[Tensor, Tuple[float, float, float]]:
    """
    Simplified MACLO-style unified loss.

    Args:
        Z_shared:
            Shared feature tensor that connects all tasks
            (requires_grad=True), e.g. [B, C, H, W] or [B, D].
        losses:
            Tuple (L_seg, L_cls, L_age): scalar loss tensors.
        alpha:
            Temperature controlling the influence of cosine similarities
            (synergy) between task gradients.
        beta:
            Temperature controlling the influence of gradient norms
            (difficulty-aware weighting).

    Returns:
        L_maclo:
            Unified scalar loss (weighted combination of the three tasks).
        lambdas:
            Tuple of (λ_seg, λ_cls, λ_age) as Python floats, useful for logging.
    """

    L_seg, L_cls, L_age = losses

    # Ensure Z_shared is part of the computation graph
    assert Z_shared.requires_grad, "Z_shared must require grad for MACLO weighting."

    # ---- Compute gradients of each task w.r.t Z_shared ----
    grads = []
    for L in (L_seg, L_cls, L_age):
        g = torch.autograd.grad(
            L,
            Z_shared,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if g is None:
            # If a task does not depend on Z_shared, use zero gradient
            g = torch.zeros_like(Z_shared)
        grads.append(g.reshape(-1))  # flatten to [N]

    g_s, g_c, g_a = grads  # seg, cls, age

    def cos_sim(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
        num = (x * y).sum()
        den = x.norm() * y.norm() + eps
        return num / den

    # ---- Synergy (cosine similarities between task gradients) ----
    s_sc = cos_sim(g_s, g_c)
    s_sa = cos_sim(g_s, g_a)
    s_ca = cos_sim(g_c, g_a)

    # We can interpret these as how "aligned" the tasks are
    s_vec = torch.stack([s_sc, s_sa, s_ca])      # [3]
    s_pos = torch.exp(alpha * s_vec)             # softmax-like scaling
    s_norm = s_pos / (s_pos.sum() + 1e-8)        # normalized synergy scores

    # ---- Task difficulty (gradient norms) ----
    norms = torch.stack([g_s.norm(), g_c.norm(), g_a.norm()])  # [3]
    inv_norms = 1.0 / (norms + 1e-8)                           # larger for "harder" tasks
    lam_raw = torch.exp(beta * inv_norms)

    # Combine difficulty and synergy:
    #   - lam_raw from norms (difficulty)
    #   - s_norm from cosine similarity (synergy)
    lam_raw = lam_raw * s_norm
    lam = lam_raw / (lam_raw.sum() + 1e-8)                      # [3]

    lam_s, lam_c, lam_a = lam

    # ---- Unified MACLO-style loss ----
    L_maclo = lam_s * L_seg + lam_c * L_cls + lam_a * L_age

    return L_maclo, (lam_s.item(), lam_c.item(), lam_a.item())
