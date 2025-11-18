import torch.nn as nn
from .components.backbones import TwoStageFocalTransformerHopfield, ModifiedEfficientViT, SCCTFusion
from .components.blocks import TopKSparseAttention as SPAM
from .components.heads import MultiTaskHeads, UncertaintyWeights

class MACLOSiamNet(nn.Module):
    def __init__(self, c=32, spam_keep=0.1, tasks=("seg","cls","age")):
        super().__init__()
        self.mri = TwoStageFocalTransformerHopfield(c_in=1, c=c)
        self.ct  = ModifiedEfficientViT(c_in=1, c=c)
        self.fuse = SCCTFusion(c=c)
        self.spam = SPAM(c, keep=spam_keep)
        self.heads = MultiTaskHeads(c=c)
        self.uw = UncertaintyWeights(tasks=tasks)
    def forward(self, mri, ct, metadata=None):
        fm = self.mri(mri)
        fc = self.ct(ct)
        z  = self.fuse(fm, fc, metadata)
        z  = self.spam(z)
        seg, cls, age = self.heads(z)
        return seg, cls, age
