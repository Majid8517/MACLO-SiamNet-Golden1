from __future__ import annotations
from typing import Dict, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SparsePeerAttention(nn.Module):
    """
    SPAM v2: task-specific sparse attention over a compact spatial token grid.

    Each task owns a learnable query. Only the top-k token scores are retained,
    yielding explicit task-specific sparse context without full-resolution attention.
    """
    def __init__(
        self,
        channels: int,
        tasks: List[str],
        keep_ratio: float = 0.25,
        pool_size: int = 8,
    ):
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1].")
        self.tasks = list(tasks)
        self.keep_ratio = keep_ratio
        self.pool_size = pool_size
        self.task_queries = nn.Parameter(torch.randn(len(tasks), channels) * 0.02)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.refine = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            for task in self.tasks
        })

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        B, C, _, _ = x.shape
        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        key = self.key(tokens)
        value = self.value(tokens)
        scale = C ** -0.5

        outputs = {}
        for i, task in enumerate(self.tasks):
            query = self.task_queries[i].view(1, 1, C).expand(B, 1, C)
            scores = torch.matmul(query, key.transpose(1, 2)).squeeze(1) * scale
            k = max(1, int(scores.shape[-1] * self.keep_ratio))
            top_values, top_indices = scores.topk(k, dim=-1)

            sparse_scores = torch.full_like(scores, float("-inf"))
            sparse_scores.scatter_(1, top_indices, top_values)
            attention = torch.softmax(sparse_scores, dim=-1)
            context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
            outputs[task] = self.refine[task](context)

        return outputs
