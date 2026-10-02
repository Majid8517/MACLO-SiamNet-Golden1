from __future__ import annotations
from typing import Dict, Iterable, List
import torch
from torch import nn, Tensor


def _flatten(grads, params):
    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            grad = torch.zeros_like(param)
        pieces.append(grad.reshape(-1))
    return torch.cat(pieces)


def _unflatten(vector: Tensor, params: List[nn.Parameter]):
    out, offset = [], 0
    for param in params:
        n = param.numel()
        out.append(vector[offset:offset+n].view_as(param))
        offset += n
    return out


class MACLOController:
    """
    Gradient-level MACLO v2.

    For the designated shared parameter set:
      1. computes one gradient vector per active task;
      2. logs pairwise cosine affinity;
      3. attenuates negatively aligned peer components;
      4. balances adjusted task gradients by gradient norm;
      5. writes one unified gradient back to the shared parameters.

    Non-shared/task-head parameters retain their ordinary gradients from
    the summed active task losses.
    """
    def __init__(
        self,
        conflict_strength: float = 1.0,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        self.conflict_strength = conflict_strength
        self.temperature = temperature
        self.eps = eps

    def task_gradients(
        self,
        losses: Dict[str, Tensor],
        shared_params: Iterable[nn.Parameter],
    ):
        params = [p for p in shared_params if p.requires_grad]
        tasks = list(losses.keys())
        if len(tasks) == 0:
            raise ValueError("No active task losses were provided.")

        rows = []
        for task in tasks:
            grads = torch.autograd.grad(
                losses[task],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            rows.append(_flatten(grads, params))

        G = torch.stack(rows, dim=0)
        norms = G.norm(dim=1).clamp_min(self.eps)
        cosine = (G @ G.t()) / (norms[:, None] * norms[None, :])
        return tasks, params, G, norms, cosine

    def overwrite_shared_gradients(
        self,
        losses: Dict[str, Tensor],
        shared_params: Iterable[nn.Parameter],
    ):
        tasks, params, G, norms, cosine = self.task_gradients(losses, shared_params)

        adjusted = []
        for i in range(len(tasks)):
            gi = G[i].clone()
            for j in range(len(tasks)):
                if i == j:
                    continue
                cij = cosine[i, j]
                if cij.item() < 0.0:
                    gj = G[j]
                    projection = (gi @ gj) / (gj @ gj + self.eps) * gj
                    gi = gi - self.conflict_strength * (-cij).detach() * projection
            adjusted.append(gi)

        A = torch.stack(adjusted, dim=0)
        adjusted_norms = A.norm(dim=1).clamp_min(self.eps)
        target_norm = adjusted_norms.mean().detach()
        balance = target_norm / adjusted_norms
        weights = torch.softmax(balance / self.temperature, dim=0)
        unified = (weights[:, None] * A).sum(dim=0)

        for param, grad in zip(params, _unflatten(unified, params)):
            param.grad = grad.clone()

        eye = torch.eye(len(tasks), dtype=torch.bool, device=cosine.device)
        offdiag = ~eye
        negative_fraction = (
            ((cosine < 0) & offdiag).float().sum()
            / max(1, len(tasks) * (len(tasks) - 1))
        )

        return {
            "tasks": tasks,
            "cosine_matrix": cosine.detach(),
            "gradient_norms": norms.detach(),
            "weights": weights.detach(),
            "negative_pair_fraction": negative_fraction.detach(),
        }
