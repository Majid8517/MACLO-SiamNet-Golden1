import torch.nn as nn
import torch

class MultiTaskHeads(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.seg = nn.Conv2d(c, 1, 1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 2))
        self.age = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 1))
    def forward(self, z):
        return self.seg(z), self.cls(z), self.age(z)

class UncertaintyWeights(nn.Module):
    def __init__(self, tasks=("seg","cls","age")):
        super().__init__()
        self.tasks = tasks
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in tasks})
    def forward(self, losses):
        total = 0.0
        for t, L in losses.items():
            s = self.log_vars[t]
            total = total + (L * torch.exp(-s) + s)
        return total
