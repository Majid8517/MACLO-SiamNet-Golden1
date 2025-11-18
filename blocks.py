import torch, torch.nn as nn
from einops import rearrange

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

class SimpleAttention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Conv2d(c, c, 1)
        self.to_k = nn.Conv2d(c, c, 1)
        self.to_v = nn.Conv2d(c, c, 1)
        self.proj = nn.Conv2d(c, c, 1)
    def forward(self, x):
        B,C,H,W = x.shape; h = self.heads; d = C//h
        q = rearrange(self.to_q(x), "b (h d) h1 w1 -> b h d (h1 w1)", h=h, d=d)
        k = rearrange(self.to_k(x), "b (h d) h1 w1 -> b h d (h1 w1)", h=h, d=d)
        v = rearrange(self.to_v(x), "b (h d) h1 w1 -> b h d (h1 w1)", h=h, d=d)
        attn = (q.transpose(2,3) @ k) / (d**0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v.transpose(2,3)
        out = out.transpose(2,3)
        out = rearrange(out, "b h d (n) -> b (h d) 1 n", h=h, d=d, n=H*W)
        out = out.view(B, C, H, W)
        return self.proj(out)

class HopfieldLike(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.Sigmoid())
    def forward(self, x):
        m = self.gate(x)
        return x * (0.5 + 0.5*m)

class DeformableLike(nn.Module):
    def __init__(self, c, dilations=(1,2,3)):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseSeparableConv(c, c, k=3, s=1, p=d) for d in dilations])
        self.fuse = nn.Conv2d(c*len(dilations), c, 1)
    def forward(self, x):
        import torch
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1))

class TopKSparseAttention(nn.Module):
    def __init__(self, c, keep=0.1):
        super().__init__()
        self.keep = keep
        self.q = nn.Conv2d(c, c, 1); self.k = nn.Conv2d(c, c, 1); self.v = nn.Conv2d(c, c, 1)
    def forward(self, x):
        import torch
        B,C,H,W = x.shape; N = H*W
        q = self.q(x).flatten(2).transpose(1,2)
        k = self.k(x).flatten(2).transpose(1,2)
        v = self.v(x).flatten(2).transpose(1,2)
        scores = (q @ k.transpose(1,2)) / (C**0.5)
        k_keep = max(1, int(self.keep*N))
        topk_vals, topk_idx = scores.topk(k_keep, dim=-1)
        mask = scores.new_full(scores.shape, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        attn = mask.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B,C,H,W)
        return out
