from __future__ import annotations
import torch
from torch import nn, Tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, drop_path: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.layer_scale = nn.Parameter(torch.ones(channels) * 1e-6)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x * self.layer_scale.view(1, -1, 1, 1)
        return residual + self.drop_path(x)


class ImageEncoderV3(nn.Module):
    """
    Compact hierarchical encoder with stochastic depth.
    Output scales are 1/4, 1/8, 1/16 and 1/32 of the input resolution.
    """
    def __init__(
        self,
        in_channels: int = 1,
        channels=(32, 64, 128, 256),
        depths=(2, 2, 3, 3),
        max_drop_path: float = 0.15,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        total = sum(depths)
        rates = torch.linspace(0, max_drop_path, total).tolist()
        cursor = 0

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c1, 3, 2),
            ConvNormAct(c1, c1, 3, 2),
        )

        def stage(c, depth):
            nonlocal cursor
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtLiteBlock(c, drop_path=rates[cursor]))
                cursor += 1
            return nn.Sequential(*blocks)

        self.stage1 = stage(c1, depths[0])
        self.down2 = ConvNormAct(c1, c2, 3, 2)
        self.stage2 = stage(c2, depths[1])
        self.down3 = ConvNormAct(c2, c3, 3, 2)
        self.stage3 = stage(c3, depths[2])
        self.down4 = ConvNormAct(c3, c4, 3, 2)
        self.stage4 = stage(c4, depths[3])

    def forward(self, x: Tensor):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        return [f1, f2, f3, f4]
