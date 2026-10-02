from __future__ import annotations
import torch
from torch import nn, Tensor


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualDWBlock(nn.Module):
    """Memory-efficient ConvNeXt-like residual block."""
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.act(self.pw1(x))
        x = self.pw2(x)
        return x + residual


class ModalityStem(nn.Module):
    """Per-modality adapter before the shared feature hierarchy."""
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(1, out_channels, 3, 2),
            ConvNormAct(out_channels, out_channels, 3, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SharedEncoder(nn.Module):
    """Shared hierarchy after modality-specific intensity adaptation."""
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stage1 = nn.Sequential(ResidualDWBlock(c1), ResidualDWBlock(c1))
        self.down2 = ConvNormAct(c1, c2, 3, 2)
        self.stage2 = nn.Sequential(ResidualDWBlock(c2), ResidualDWBlock(c2))
        self.down3 = ConvNormAct(c2, c3, 3, 2)
        self.stage3 = nn.Sequential(ResidualDWBlock(c3), ResidualDWBlock(c3))
        self.down4 = ConvNormAct(c3, c4, 3, 2)
        self.stage4 = nn.Sequential(ResidualDWBlock(c4), ResidualDWBlock(c4))

    def forward(self, x: Tensor):
        f1 = self.stage1(x)              # 1/4
        f2 = self.stage2(self.down2(f1)) # 1/8
        f3 = self.stage3(self.down3(f2)) # 1/16
        f4 = self.stage4(self.down4(f3)) # 1/32
        return [f1, f2, f3, f4]
