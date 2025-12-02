"""
blocks.py

Core building blocks used in MACLO-SiamNet backbones:

- ConvBNAct:        Conv2d + BatchNorm + ReLU
- DepthwiseSeparableConv: depthwise + pointwise conv (EfficientViT style)
- SimpleAttention:  lightweight multi-head self-attention on 2D feature maps
- HopfieldLike:     simple channel-wise attractor-style gating
- DeformableLike:   multi-branch dilated depthwise conv + 1x1 fusion
- TopKSparseAttention: sparse token attention keeping top-k scores only
"""

from __future__ import annotations

from typing import Sequence

import math

import torch
from torch import nn, Tensor
from einops import rearrange


__all__ = [
    "ConvBNAct",
    "DepthwiseSeparableConv",
    "SimpleAttention",
    "HopfieldLike",
    "DeformableLike",
    "TopKSparseAttention",
]


class ConvBNAct(nn.Module):
    """
    Basic conv block: Conv2d + BatchNorm2d + ReLU.

    Args:
        c_in:  input channels
        c_out: output channels
        k:     kernel size
        s:     stride
        p:     padding
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution:
        depthwise (per-channel) conv + pointwise (1x1) projection.

    Args:
        c_in:     input channels
        c_out:    output channels
        k:        kernel size
        s:        stride
        p:        padding
        dilation: dilation rate (used by DeformableLike branches)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=dilation,
            groups=c_in,
            bias=False,
        )
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SimpleAttention(nn.Module):
    """
    Lightweight multi-head self-attention over 2D feature maps.

    Input:  [B, C, H, W]  with C divisible by `heads`
    Output: [B, C, H, W]

    Args:
        c:      number of channels
        heads:  number of attention heads
    """

    def __init__(self, c: int, heads: int = 4) -> None:
        super().__init__()
        assert c % heads == 0, "channels must be divisible by number of heads"
        self.heads = heads
        self.head_dim = c // heads

        self.to_q = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h, d = self.heads, self.head_dim
        N = H * W

        # [B, C, H, W] -> [B, h, N, d]
        q = rearrange(self.to_q(x), "b (h d) h1 w1 -> b h (h1 w1) d", h=h, d=d)
        k = rearrange(self.to_k(x), "b (h d) h1 w1 -> b h (h1 w1) d", h=h, d=d)
        v = rearrange(self.to_v(x), "b (h d) h1 w1 -> b h (h1 w1) d", h=h, d=d)

        # Attention: [B, h, N, N]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        attn = attn.softmax(dim=-1)

        # [B, h, N, d]
        out = attn @ v

        # Back to [B, C, H, W]
        out = rearrange(
            out,
            "b h (h1 w1) d -> b (h d) h1 w1",
            h1=H,
            w1=W,
            h=h,
            d=d,
        )
        out = self.proj(out)
        return out


class HopfieldLike(nn.Module):
    """
    Simple Hopfield-like associative gating.

    Implements a channel-wise gate using global average pooling followed by
    a 1x1 conv and sigmoid, then blends the original features with:

        x * (0.5 + 0.5 * gate)

    This roughly mimics attractor-style stabilization of salient patterns.
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        m = self.gate(x)            # [B, C, 1, 1]
        return x * (0.5 + 0.5 * m)  # mildly gated residual


class DeformableLike(nn.Module):
    """
    Deformable-like multi-branch convolution.

    Approximates deformable behavior using multiple dilated
    DepthwiseSeparableConv branches with different dilation rates.

    Args:
        c:         number of channels
        dilations: dilation rates for each branch
    """

    def __init__(self, c: int, dilations: Sequence[int] = (1, 2, 3)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv(
                    c,
                    c,
                    k=3,
                    s=1,
                    p=d,
                    dilation=d,
                )
                for d in dilations
            ]
        )
        self.fuse = nn.Conv2d(c * len(dilations), c, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        feats = [b(x) for b in self.branches]  # list of [B, C, H, W]
        x = torch.cat(feats, dim=1)            # [B, C * n_branches, H, W]
        x = self.fuse(x)                       # [B, C, H, W]
        return x


class TopKSparseAttention(nn.Module):
    """
    Top-k sparse self-attention on flattened tokens.

    Keeps only the top `keep` fraction of attention scores for each query
    and sets the rest to -inf before softmax.

    Args:
        c:    number of channels
        keep: fraction of spatial positions to keep (0 < keep <= 1)
    """

    def __init__(self, c: int, keep: float = 0.1) -> None:
        super().__init__()
        assert 0 < keep <= 1.0, "`keep` must be in (0, 1]"
        self.keep = keep

        self.q = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.k = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.v = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W

        # [B, C, H, W] -> [B, N, C]
        q = self.q(x).flatten(2).transpose(1, 2)  # [B, N, C]
        k = self.k(x).flatten(2).transpose(1, 2)  # [B, N, C]
        v = self.v(x).flatten(2).transpose(1, 2)  # [B, N, C]

        # [B, N, N]
        scores = (q @ k.transpose(1, 2)) / math.sqrt(C)

        k_keep = max(1, int(self.keep * N))

        # Top-k per query
        topk_vals, topk_idx = scores.topk(k_keep, dim=-1)  # [B, N, k]

        mask = scores.new_full(scores.shape, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)

        attn = mask.softmax(dim=-1)          # [B, N, N]
        out = attn @ v                       # [B, N, C]

        # Back to [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out
