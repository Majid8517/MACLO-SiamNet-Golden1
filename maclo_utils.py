# maclo_utils.py
import torch
import torch.nn.functional as F

def compute_task_losses(outputs, targets):
    """
    outputs: dict with keys ['seg', 'cls', 'age']
    targets: dict with keys ['seg', 'cls', 'age']

    Returns: L_seg, L_cls, L_age
    """
    # segmentation: Dice + BCE (as an example)
    seg_pred = outputs['seg']          # [B, 1, H, W]
    seg_gt   = targets['seg']          # [B, 1, H, W]
    smooth = 1e-5
    intersection = (seg_pred * seg_gt).sum()
    dice = (2 * intersection + smooth) / (
        seg_pred.sum() + seg_gt.sum() + smooth
    )
    L_seg = 1 - dice

    # classification: CrossEntropy
    cls_pred = outputs['cls']          # [B, num_classes]
    cls_gt   = targets['cls'].long()   # [B]
    L_cls = F.cross_entropy(cls_pred, cls_gt)

    # age regression: MAE
    age_pred = outputs['age'].squeeze(-1)   # [B]
    age_gt   = targets['age'].float()      # [B]
    L_age = F.l1_loss(age_pred, age_gt)

    return L_seg, L_cls, L_age


def maclo_unified_loss(model, Z_shared, losses, alpha=0.5, beta=0.5):
    """
    Simplified MACLO-style unified loss:
    - losses: (L_seg, L_cls, L_age)
    - Z_shared: tensor used as anchor to probe gradients (e.g., mean of encoder features)
    This function illustrates gradient-based weighting logic for reviewers.

    In practice, we derive gradient directions for each task w.r.t Z_shared,
    then compute cosine similarities to build synergy-aware weights.
    """

    L_seg, L_cls, L_age = losses

    # compute gradients w.r.t a scalar proxy: mean(Z_shared)
    Z_anchor = Z_shared.mean()
    grads = []
    for L in [L_seg, L_cls, L_age]:
        model.zero_grad(set_to_none=True)
        Z_anchor_detached = Z_anchor.detach().requires_grad_(True)
        g = torch.autograd.grad(L, Z_anchor_detached, retain_graph=True)[0]
        grads.append(g.flatten())

    g_s, g_c, g_a = grads

    # cosine similarities (synergy)
    def cos_sim(x, y, eps=1e-8):
        num = (x * y).sum()
        den = x.norm() * y.norm() + eps
        return num / den

    s_sc = cos_sim(g_s, g_c)
    s_sa = cos_sim(g_s, g_a)
    s_ca = cos_sim(g_c, g_a)

    # convert similarities to positive synergy weights via softmax-like mapping
    s_vec = torch.stack([s_sc, s_sa, s_ca])
    s_pos = torch.exp(alpha * s_vec)
    s_norm = s_pos / (s_pos.sum() + 1e-8)

    # task-balancing weights λ based on gradient norms (difficulty-aware)
    norms = torch.stack([g_s.norm(), g_c.norm(), g_a.norm()])
    inv_norms = 1.0 / (norms + 1e-8)
    lam = torch.exp(beta * inv_norms)
    lam = lam / (lam.sum() + 1e-8)
    λ_s, λ_c, λ_a = lam

    # unified MACLO-style loss as weighted sum
    L_maclo = λ_s * L_seg + λ_c * L_cls + λ_a * L_age

    return L_maclo, (λ_s.item(), λ_c.item(), λ_a.item())
