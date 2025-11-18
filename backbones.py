import torch.nn as nn
from .blocks import ConvBNAct, SimpleAttention, HopfieldLike, DepthwiseSeparableConv, DeformableLike

class TwoStageFocalTransformerHopfield(nn.Module):
    def __init__(self, c_in=1, c=32):
        super().__init__()
        self.stage1 = nn.Sequential(ConvBNAct(c_in, c), SimpleAttention(c), HopfieldLike(c))
        self.stage2 = nn.Sequential(ConvBNAct(c, c), SimpleAttention(c), HopfieldLike(c))
    def forward(self, x):
        x = self.stage1(x); x = self.stage2(x); return x

class ModifiedEfficientViT(nn.Module):
    def __init__(self, c_in=1, c=32):
        super().__init__()
        self.stem = DepthwiseSeparableConv(c_in, c)
        self.enc  = nn.Sequential(DepthwiseSeparableConv(c, c), SimpleAttention(c))
    def forward(self, x):
        x = self.stem(x); x = self.enc(x); return x

class SCCTFusion(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.reduce = nn.Conv2d(c, c, 1)
        self.mix = DeformableLike(c)
    def forward(self, fmri, fct, metadata=None):
        z = self.reduce(fmri + fct)
        z = self.mix(z)
        return z
