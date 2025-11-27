"""
MACLO-SiamNet: Reference PyTorch Implementation
-----------------------------------------------

This file provides a *reference* implementation of the MACLO-SiamNet framework
proposed for multimodal AIS analysis (MRI + CT + clinical metadata) with:

- Novel Dual-Stage NeuroFocal Transformer (NF-Transformer) for MRI
- Novel Spatio-Temporal Graph-Enhanced EfficientViT (STGE-EfficientViT) [simplified]
- Soft-Compact Convolutional Transformer (SCCT) for multimodal fusion
- Sparsely-Handled Peer Attention Module (SPAM) for cross-task refinement
- MACLO: Multi-Task Adaptive Co-Learning Optimization (gradient-level)



Dependencies:
    - PyTorch >= 1.12
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ============================================================
# 1. NeuroFocal Transformer (MRI Encoder)
# ============================================================

class NeuroFocalNorm(nn.Module):
    """
    NeuroFocal Adaptive Normalization (NF-Norm)

    Implements:
        z_tilde_i = LayerNorm(z_i) * (1 + gamma * NF(z_i))
        NF(z_i)   = ||z_i - mu_i||_1 / (sigma_i + eps)

    where z has shape (B, N, D) – tokens per slice.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, gamma: float = 1.0):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.norm = nn.LayerNorm(num_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, D)
        z_norm = self.norm(z)                           # (B,N,D)
        mu = z_norm.mean(dim=-1, keepdim=True)          # (B,N,1)
        var = ((z_norm - mu) ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(var + self.eps)              # (B,N,1)

        nf = (z_norm - mu).abs().sum(dim=-1, keepdim=True) / (sigma + self.eps)
        z_tilde = z_norm * (1.0 + self.gamma * nf)
        return z_tilde


class AttractorGuidedNeuroFocalAttention(nn.Module):
    """
    Attractor-Guided NeuroFocal Attention (AG-NFA)

    Multi-head attention with an additional penalty term:

        A = softmax( QK^T / sqrt(d_k) - alpha * Ω(Q,K) )

    where Ω(Q,K) ~ ||Q - K||^2 / (exp(-delta * NF(Q)) + 1),
    and NF(Q) is a NeuroFocal-like deviation term along the embedding dimension.
    """

    def __init__(self, dim: int, num_heads: int = 4, alpha: float = 1.0, delta: float = 1.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.alpha = alpha
        self.delta = delta
        self.eps = 1e-5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, H, N, Dh)
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, N, Dh) -> (B, N, D)
        B, H, N, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, N, H * Dh)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, D)
        Q = self._split_heads(self.q_proj(z))  # (B,H,N,Dh)
        K = self._split_heads(self.k_proj(z))
        V = self._split_heads(self.v_proj(z))

        d_k = math.sqrt(self.head_dim)

        # basic scaled dot-product attention logits
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # (B,H,N,N)

        # NF(Q) along embedding dimension
        mu_q = Q.mean(dim=-1, keepdim=True)
        var_q = ((Q - mu_q) ** 2).mean(dim=-1, keepdim=True)
        sigma_q = torch.sqrt(var_q + self.eps)
        nf_q = (Q - mu_q).abs().sum(dim=-1, keepdim=True) / (sigma_q + self.eps)  # (B,H,N,1)

        # penalty Ω(Q,K) ~ ||Q-K||^2 / (exp(-delta NF(Q)) + 1)
        diff = Q.unsqueeze(-2) - K.unsqueeze(-3)  # (B,H,N,N,Dh)
        omega = (diff ** 2).sum(dim=-1)          # (B,H,N,N)
        omega = omega / (torch.exp(-self.delta * nf_q) + 1.0)

        attn_logits = attn_logits - self.alpha * omega
        A = F.softmax(attn_logits, dim=-1)       # (B,H,N,N)
        out = torch.matmul(A, V)                 # (B,H,N,Dh)
        out = self._merge_heads(out)             # (B,N,D)
        return self.out_proj(out)


class LesionEnergyPatchFusion(nn.Module):
    """
    Lesion-Energy Patch Fusion (LE-Fusion)

    Groups tokens in sets of group_size and fuses them via
        E_i = ||Y_i||_2
        λ_i = softmax(beta * E_i)
        Z_merge = Σ λ_i Y_i
    """

    def __init__(self, beta: float = 1.0, group_size: int = 4):
        super().__init__()
        self.beta = beta
        self.group_size = group_size

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Y: (B, N, D), assume N % group_size == 0
        """
        B, N, D = Y.shape
        G = N // self.group_size
        Y_grouped = Y.view(B, G, self.group_size, D)   # (B,G,group_size,D)

        E = torch.norm(Y_grouped, p=2, dim=-1)        # (B,G,group_size)
        logits = self.beta * E
        lam = F.softmax(logits, dim=-1).unsqueeze(-1) # (B,G,group_size,1)

        Z_merge = (lam * Y_grouped).sum(dim=2)        # (B,G,D)
        return Z_merge


class NeuroFocalTransformerMRI(nn.Module):
    """
    NeuroFocal Transformer MRI encoder:

        Z_MRI = LE-Fusion( AG-NFA( NF-Norm( PatchEmbed(I_MRI) ) ) )

    For simplicity, MRI input is a single slice (B, C, H, W) with H=W=IMG_SIZE.
    """

    def __init__(self, in_channels=1, patch_size=16, emb_dim=256, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.img_size = img_size

        # Non-overlapping patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.nf_norm = NeuroFocalNorm(emb_dim)
        self.ag_nfa = AttractorGuidedNeuroFocalAttention(dim=emb_dim, num_heads=4)
        self.le_fusion = LesionEnergyPatchFusion(beta=1.0, group_size=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) MRI slice
        returns: Z_mri (B, N_mri, D)
        """
        z = self.patch_embed(x)                # (B,D,H',W')
        z = z.flatten(2).transpose(1, 2)       # (B,N,D)
        z = self.nf_norm(z)
        z = self.ag_nfa(z)
        z = self.le_fusion(z)                  # (B,N',D)
        return z


# ============================================================
# 2. Simplified STGE-EfficientViT (CT Encoder)
# ============================================================

class AdaptiveTemporalContrastEnhancement(nn.Module):
    """
    Simplified ATCE for CT slices (treated as a temporal sequence).

    X: (B, T, N, D)
    T_i = softmax( Q_i K_i^T / sqrt(d_k) + eta * ΔC_i )
    ΔC_i = ||X_i - X_{i-1}||_1 (approx. over tokens)
    """

    def __init__(self, dim: int, num_heads: int = 4, eta: float = 1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eta = eta

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x):
        B, T, N, D = x.shape
        x = x.view(B, T, N, self.num_heads, self.head_dim)
        return x.permute(0, 1, 3, 2, 4)  # (B,T,H,N,Dh)

    def _merge_heads(self, x):
        B, T, H, N, Dh = x.shape
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(B, T, N, H * Dh)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B,T,N,D)
        Q = self._split_heads(self.q_proj(X))
        K = self._split_heads(self.k_proj(X))
        V = self._split_heads(self.v_proj(X))

        d_k = math.sqrt(self.head_dim)
        logits = torch.matmul(Q, K.transpose(-2, -1)) / d_k   # (B,T,H,N,N)

        # temporal contrast prior over mean tokens
        X_mean = X.mean(dim=2)  # (B,T,D)
        diff = (X_mean[:, 1:] - X_mean[:, :-1]).abs().sum(dim=-1, keepdim=True)  # (B,T-1,1)
        pad = torch.zeros_like(diff[:, 0:1])
        delta_c = torch.cat([pad, diff], dim=1)   # (B,T,1)
        delta_c = delta_c.unsqueeze(2).unsqueeze(-1)  # (B,T,1,1,1)

        logits = logits + self.eta * delta_c
        A = F.softmax(logits, dim=-1)
        Z = torch.matmul(A, V)  # (B,T,H,N,Dh)
        Z = self._merge_heads(Z)  # (B,T,N,D)
        return self.out_proj(Z)


class RegionalAnatomicalGraphAttention(nn.Module):
    """
    Simplified RAG-Former component:

    Given node features h (e.g., slice- or patch-level), and an anatomical
    distance matrix dist_mat, performs distance-aware graph attention:

        G_ij ∝ exp( (Wh_i)^T (Wh_j) + lambda * Sim_anat(i,j) )
        Sim_anat(i,j) = exp( -d_ij / sigma )
    """

    def __init__(self, dim: int, lambda_anat: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.lambda_anat = lambda_anat
        self.sigma = sigma

    def forward(self, h: torch.Tensor, dist_mat: torch.Tensor) -> torch.Tensor:
        """
        h: (B, N, D)   node features
        dist_mat: (N, N)  anatomical distances
        """
        B, N, D = h.shape
        Wh = self.proj(h)                         # (B,N,D)
        sim = torch.matmul(Wh, Wh.transpose(1, 2))  # (B,N,N)

        sim_anat = torch.exp(-dist_mat / self.sigma).to(h.device)  # (N,N)
        sim_anat = sim_anat.unsqueeze(0)                           # (1,N,N)

        logits = sim + self.lambda_anat * sim_anat
        G = F.softmax(logits, dim=-1)              # (B,N,N)
        Z = torch.matmul(G, h)                     # (B,N,D)
        return Z


class STGEEfficientViTEncoder(nn.Module):
    """
    Simplified STGE-EfficientViT encoder for CT:

        1) Patch embedding for each slice
        2) ATCE over temporal dimension (slices)
        3) RAG-Former over patch graph
        4) Lightweight FFN

    Returns a CT representation Z_ct: (B, N_ct, D)
    """

    def __init__(self, in_channels=1, emb_dim=256, img_size=256,
                 patch_size=16, num_slices=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_slices = num_slices

        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.atce = AdaptiveTemporalContrastEnhancement(dim=emb_dim, num_heads=4, eta=1.0)
        self.rag = RegionalAnatomicalGraphAttention(dim=emb_dim, lambda_anat=1.0, sigma=1.0)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim * 4, emb_dim)
        )

        # Precompute a simple 1D anatomical distance matrix over patches
        num_patches = (img_size // patch_size) ** 2
        coords = torch.stack(torch.meshgrid(
            torch.arange(img_size // patch_size),
            torch.arange(img_size // patch_size),
            indexing='ij'
        ), dim=-1).view(-1, 2).float()   # (N,2)
        dists = torch.cdist(coords, coords, p=2)     # (N,N)
        self.register_buffer("dist_mat", dists)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) CT volume
        returns: Z_ct (B, N_ct, D)
        """
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.patch_embed(x[:, t])         # (B,D,H',W')
            f = f.flatten(2).transpose(1, 2)      # (B,N,D)
            feats.append(f)
        X_seq = torch.stack(feats, dim=1)         # (B,T,N,D)

        Z_atce = self.atce(X_seq)                 # (B,T,N,D)
        # collapse temporal dimension by averaging
        Z_flat = Z_atce.mean(dim=1)               # (B,N,D)

        Z_rag = self.rag(Z_flat, self.dist_mat)   # (B,N,D)
        Z_ffn = self.ffn(Z_rag)                   # (B,N,D)

        return Z_ffn


# ============================================================
# 3. SCCT: Multimodal Fusion (MRI + CT + Metadata)
# ============================================================

class SoftPool1D(nn.Module):
    """
    Soft pooling over sequence dimension:
        SoftPool(X) = Σ X[i] * softmax(||X[i]||) / Σ softmax(||X[i]||)
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, D)
        """
        logits = X.norm(p=2, dim=-1, keepdim=True)  # (B,N,1)
        weights = F.softmax(logits, dim=1)          # (B,N,1)
        pooled = (X * weights).sum(dim=1)           # (B,D)
        return pooled


class SCCTFusion(nn.Module):
    """
    Soft-Compact Convolutional Transformer (SCCT) – simplified:

        1) Concatenate MRI and CT tokens along sequence dimension.
        2) Append metadata context token.
        3) Transformer encoder for cross-modal self-attention.
        4) Soft pooling over tokens.
        5) MLP projection to shared latent representation Z.

    """

    def __init__(self, emb_dim=256, meta_dim=64, num_layers=2, num_heads=4):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.softpool = SoftPool1D()

        self.meta_proj = nn.Linear(meta_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, Z_mri: torch.Tensor, Z_ct: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Z_mri: (B, N_m, D)
        Z_ct:  (B, N_c, D)
        meta:  (B, meta_dim)
        returns:
            Z: shared latent representation (B, D)
        """
        B, _, D = Z_mri.shape
        # concat MRI + CT tokens
        tokens = torch.cat([Z_mri, Z_ct], dim=1)      # (B, N_m+N_c, D)

        # metadata as extra context token
        meta_tok = self.meta_proj(meta).unsqueeze(1)  # (B,1,D)
        tokens = torch.cat([meta_tok, tokens], dim=1) # (B,1+N_m+N_c,D)

        tokens_enc = self.encoder(tokens)             # (B,N_total,D)
        pooled = self.softpool(tokens_enc)            # (B,D)
        Z = self.mlp(pooled)                          # (B,D)
        return Z


# ============================================================
# 4. SPAM: Sparsely-Handled Peer Attention Module
# ============================================================

class SPAModule(nn.Module):
    """
    Sparsely-Handled Peer Attention Module (SPAM)

    - L1-guided soft thresholding (sparsity)
    - Top-K feature selection across channels
    - Peer multi-head attention across task-specific branches
    """

    def __init__(self, dim: int, num_heads: int = 4, tau: float = 1e-3, topk: int = 64):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tau = tau
        self.topk = topk

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def soft_threshold(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.clamp(x.abs() - self.tau, min=0.0)

    def _mh_attention(self, Q, K, V):
        """
        Q,K,V: (B, N, D)
        returns: (B, N, D)
        """
        B, N, D = Q.shape
        H = self.num_heads
        Dh = self.head_dim

        def split_heads(x):
            x = x.view(B, N, H, Dh)
            return x.permute(0, 2, 1, 3)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        d_k = math.sqrt(Dh)
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) / d_k
        A = F.softmax(logits, dim=-1)
        out = torch.matmul(A, Vh)  # (B,H,N,Dh)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.out_proj(out)

    def forward(self, F_cls: torch.Tensor, F_seg: torch.Tensor, F_age: torch.Tensor):
        """
        F_cls, F_seg, F_age: (B, N, D)
        """
        # 1) Concatenate and apply L1 soft-thresholding
        X = torch.cat([F_cls, F_seg, F_age], dim=1)  # (B,3N,D)
        X_sparse = self.soft_threshold(X)

        # 2) Top-K feature selection over channels (approx. representation)
        B, TN, D = X_sparse.shape
        saliency = X_sparse.abs().mean(dim=1)        # (B,D)
        k = min(self.topk, D)
        topk_vals, topk_idx = torch.topk(saliency, k, dim=-1)
        mask = torch.zeros_like(saliency, dtype=torch.bool)
        for b in range(B):
            mask[b, topk_idx[b]] = True
        mask = mask.unsqueeze(1).expand(B, TN, D)
        X_topk = X_sparse * mask

        # split back into tasks
        N = F_cls.size(1)
        F_cls_s = X_topk[:, 0:N]
        F_seg_s = X_topk[:, N:2 * N]
        F_age_s = X_topk[:, 2 * N:3 * N]

        K_all = self.k_proj(X_topk)
        V_all = self.v_proj(X_topk)

        Q_cls = self.q_proj(F_cls_s)
        Q_seg = self.q_proj(F_seg_s)
        Q_age = self.q_proj(F_age_s)

        F_cls_ref = self._mh_attention(Q_cls, K_all, V_all)
        F_seg_ref = self._mh_attention(Q_seg, K_all, V_all)
        F_age_ref = self._mh_attention(Q_age, K_all, V_all)

        return F_cls_ref, F_seg_ref, F_age_ref


# ============================================================
# 5. MACLO: Multi-Task Adaptive Co-Learning Optimization
# ============================================================

def maclo_compute_unified_gradient(
    Z: torch.Tensor,
    losses: Dict[str, torch.Tensor],
    create_graph: bool = False
) -> torch.Tensor:
    """
    Compute MACLO unified gradient w.r.t shared representation Z.

    losses: dict with keys ["seg", "cls", "age"]
    Z:      shared representation (B, D) or (B, N, D)

    This implements Eqs. (31)–(34) in a simplified batch-wise manner.
    """
    L_s = losses["seg"]
    L_c = losses["cls"]
    L_a = losses["age"]

    g_s = torch.autograd.grad(L_s, Z, retain_graph=True, create_graph=create_graph)[0]
    g_c = torch.autograd.grad(L_c, Z, retain_graph=True, create_graph=create_graph)[0]
    g_a = torch.autograd.grad(L_a, Z, retain_graph=True, create_graph=create_graph)[0]

    def flat(g):
        return g.reshape(g.size(0), -1)

    gs_f = flat(g_s)
    gc_f = flat(g_c)
    ga_f = flat(g_a)

    eps = 1e-8

    def cosine_affinity(g_i, g_j):
        num = (g_i * g_j).sum(dim=-1)
        den = g_i.norm(p=2, dim=-1) * g_j.norm(p=2, dim=-1) + eps
        return num / den

    # pairwise cosine similarities
    psi_sc = cosine_affinity(gs_f, gc_f)
    psi_sa = cosine_affinity(gs_f, ga_f)
    psi_cs = cosine_affinity(gc_f, gs_f)
    psi_ca = cosine_affinity(gc_f, ga_f)
    psi_as = cosine_affinity(ga_f, gs_f)
    psi_ac = cosine_affinity(ga_f, gc_f)

    def softmax_two(a, b):
        logits = torch.stack([a, b], dim=-1)
        w = F.softmax(logits, dim=-1)
        return w[..., 0], w[..., 1]

    a_sc, a_sa = softmax_two(psi_sc, psi_sa)
    a_cs, a_ca = softmax_two(psi_cs, psi_ca)
    a_as, a_ac = softmax_two(psi_as, psi_ac)

    # reshape for broadcasting
    a_sc = a_sc.view(-1, 1, 1)
    a_sa = a_sa.view(-1, 1, 1)
    a_cs = a_cs.view(-1, 1, 1)
    a_ca = a_ca.view(-1, 1, 1)
    a_as = a_as.view(-1, 1, 1)
    a_ac = a_ac.view(-1, 1, 1)

    g_s_hat = g_s + a_sc * g_c + a_sa * g_a
    g_c_hat = g_c + a_cs * g_s + a_ca * g_a
    g_a_hat = g_a + a_as * g_s + a_ac * g_c

    # task-balancing weights λ_i ~ gradient norms
    ns = gs_f.norm(p=2, dim=-1)
    nc = gc_f.norm(p=2, dim=-1)
    na = ga_f.norm(p=2, dim=-1)
    denom = ns + nc + na + eps

    lam_s = (ns / denom).view(-1, 1, 1)
    lam_c = (nc / denom).view(-1, 1, 1)
    lam_a = (na / denom).view(-1, 1, 1)

    G_maclo = lam_s * g_s_hat + lam_c * g_c_hat + lam_a * g_a_hat
    return G_maclo


# ============================================================
# 6. MACLO-SiamNet: Encoder + Heads
# ============================================================

class SegHead(nn.Module):
    """
    Simple segmentation head on top of shared latent Z.

    For demo: Z -> upsampled coarse map. In practice you would use a proper
    decoder (e.g., UNet-style).
    """

    def __init__(self, emb_dim=256, out_channels=1, img_size=256, patch_size=16):
        super().__init__()
        self.emb_dim = emb_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.out_conv = nn.Conv2d(emb_dim, out_channels, kernel_size=1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (B, D) shared latent representation
        returns: seg_logits (B,1,H,W)
        """
        B, D = Z.shape
        # broadcast to patch grid then upsample
        Z_patches = self.linear(Z).unsqueeze(-1).unsqueeze(-1)  # (B,D,1,1)
        seg = self.out_conv(Z_patches)                          # (B,1,1,1)
        seg = F.interpolate(seg, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return seg


class ClsHead(nn.Module):
    def __init__(self, emb_dim=256, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.fc(Z)


class AgeHead(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.fc(Z)


class MACLOSiamNet(nn.Module):
    """
    Full MACLO-SiamNet (simplified reference):

    MRI -> NF-Transformer
    CT  -> STGE-EfficientViT
    Fusion (MRI+CT+metadata) -> SCCT -> Z
    Z -> SegHead, ClsHead, AgeHead
    """

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 emb_dim=256,
                 meta_dim=64,
                 num_ct_slices=4,
                 num_classes=3):
        super().__init__()

        self.mri_encoder = NeuroFocalTransformerMRI(
            in_channels=1,
            patch_size=patch_size,
            emb_dim=emb_dim,
            img_size=img_size
        )

        self.ct_encoder = STGEEfficientViTEncoder(
            in_channels=1,
            emb_dim=emb_dim,
            img_size=img_size,
            patch_size=patch_size,
            num_slices=num_ct_slices
        )

        self.scct = SCCTFusion(
            emb_dim=emb_dim,
            meta_dim=meta_dim,
            num_layers=2,
            num_heads=4
        )

        self.seg_head = SegHead(emb_dim=emb_dim, out_channels=1,
                                img_size=img_size, patch_size=patch_size)
        self.cls_head = ClsHead(emb_dim=emb_dim, num_classes=num_classes)
        self.age_head = AgeHead(emb_dim=emb_dim)

    def forward(self, mri: torch.Tensor, ct: torch.Tensor, meta: torch.Tensor):
        """
        mri: (B,1,H,W)
        ct:  (B,T,1,H,W)
        meta: (B,meta_dim)
        """
        Z_mri = self.mri_encoder(mri)           # (B,Nm,D)
        Z_ct  = self.ct_encoder(ct)            # (B,Nc,D)
        Z     = self.scct(Z_mri, Z_ct, meta)   # (B,D)

        seg_logits = self.seg_head(Z)          # (B,1,H,W)
        cls_logits = self.cls_head(Z)          # (B,C)
        age_pred   = self.age_head(Z)          # (B,1)

        return Z, seg_logits, cls_logits, age_pred


# ============================================================
# 7.  Dataset 
# ============================================================

class DummyStrokeDataset(Dataset):
    """
    Dummy dataset for demonstration and sanity checks.
    Replace this with your real AIS dataset in practice.
    """

    def __init__(self, num_samples=16, img_size=256, num_ct_slices=4,
                 meta_dim=64, num_classes=3):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_ct_slices = num_ct_slices
        self.meta_dim = meta_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mri = torch.randn(1, self.img_size, self.img_size)                   # (1,H,W)
        ct  = torch.randn(self.num_ct_slices, 1, self.img_size, self.img_size)  # (T,1,H,W)
        meta = torch.randn(self.meta_dim)
        seg_gt = (torch.rand(1, self.img_size, self.img_size) > 0.9).float()    # binary mask
        cls_gt = torch.randint(0, self.num_classes, (1,)).item()
        age_gt = torch.rand(1) * 24.0   # hours, for example

        return mri, ct, meta, seg_gt, cls_gt, age_gt


# ============================================================
# 8.  MACLO
# ============================================================

def train_step(
    model: MACLOSiamNet,
    batch,
    optimizer_enc,
    optimizer_heads,
    criterion_seg,
    criterion_cls,
    criterion_age,
    device="cuda"
) -> Dict[str, float]:

    mri, ct, meta, y_seg, y_cls, y_age = batch
    mri   = mri.to(device)
    ct    = ct.to(device)
    meta  = meta.to(device)
    y_seg = y_seg.to(device)
    y_cls = y_cls.to(device)
    y_age = y_age.to(device).unsqueeze(-1)  # (B,1)

    model.train()
    optimizer_enc.zero_grad()
    optimizer_heads.zero_grad()

    # 1) Forward: encoder + heads
    Z, seg_logits, cls_logits, age_pred = model(mri, ct, meta)

    # 2) Compute task losses
    L_seg = criterion_seg(seg_logits, y_seg)
    L_cls = criterion_cls(cls_logits, y_cls)
    L_age = criterion_age(age_pred, y_age)

    losses = {"seg": L_seg, "cls": L_cls, "age": L_age}

    # 3) MACLO unified gradient w.r.t. Z
    G_maclo = maclo_compute_unified_gradient(Z, losses, create_graph=True)

    # 4) Backprop unified gradient into encoder
    Z.backward(G_maclo, retain_graph=True)

    # 5) Update heads using normal loss with detached Z
    Z_det = Z.detach()
    seg_logits_h = model.seg_head(Z_det)
    cls_logits_h = model.cls_head(Z_det)
    age_pred_h   = model.age_head(Z_det)

    L_seg_h = criterion_seg(seg_logits_h, y_seg)
    L_cls_h = criterion_cls(cls_logits_h, y_cls)
    L_age_h = criterion_age(age_pred_h, y_age)

    L_heads = L_seg_h + L_cls_h + L_age_h
    L_heads.backward()

    optimizer_enc.step()
    optimizer_heads.step()

    return {
        "L_seg": float(L_seg.detach().cpu()),
        "L_cls": float(L_cls.detach().cpu()),
        "L_age": float(L_age.detach().cpu()),
        "L_total_encoder": float((L_seg + L_cls + L_age).detach().cpu()),
        "L_heads": float(L_heads.detach().cpu())
    }


# ============================================================
# 9. Main demo (sanity check training loop)
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = 256
    patch_size = 16
    emb_dim = 256
    meta_dim = 64
    num_ct_slices = 4
    num_classes = 3

    model = MACLOSiamNet(
        img_size=img_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        meta_dim=meta_dim,
        num_ct_slices=num_ct_slices,
        num_classes=num_classes
    ).to(device)

    # Separate encoder and heads
    encoder_params = list(model.mri_encoder.parameters()) + \
                     list(model.ct_encoder.parameters()) + \
                     list(model.scct.parameters())

    head_params = list(model.seg_head.parameters()) + \
                  list(model.cls_head.parameters()) + \
                  list(model.age_head.parameters())

    optimizer_enc   = torch.optim.Adam(encoder_params, lr=1e-4)
    optimizer_heads = torch.optim.Adam(head_params,   lr=1e-4)

    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_age = nn.SmoothL1Loss()

    dataset = DummyStrokeDataset(
        num_samples=16,
        img_size=img_size,
        num_ct_slices=num_ct_slices,
        meta_dim=meta_dim,
        num_classes=num_classes
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for epoch in range(2):  # small demo
        for step, batch in enumerate(loader):
            stats = train_step(
                model,
                batch,
                optimizer_enc,
                optimizer_heads,
                criterion_seg,
                criterion_cls,
                criterion_age,
                device=device
            )
            print(f"[Epoch {epoch} | Step {step}] "
                  f"L_seg={stats['L_seg']:.4f} "
                  f"L_cls={stats['L_cls']:.4f} "
                  f"L_age={stats['L_age']:.4f} "
                  f"L_enc={stats['L_total_encoder']:.4f} "
                  f"L_heads={stats['L_heads']:.4f}")


if __name__ == "__main__":
    main()
