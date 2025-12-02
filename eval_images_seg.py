import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt as edt

from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet


# =======================================================================
# Dice Coefficient
# =======================================================================
def dice(p: np.ndarray, g: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute the Dice Similarity Coefficient between two binary masks p, g ∈ {0,1}.

    Args:
        p: predicted mask as a NumPy array.
        g: ground-truth mask as a NumPy array.
        eps: small constant to avoid division by zero.

    Returns:
        Dice score as a float in [0,1].
    """
    # Ensure binary format {0,1}
    p = (p > 0).astype("float32")
    g = (g > 0).astype("float32")

    # Intersection and normalization
    inter = (p * g).sum()
    return float((2.0 * inter + eps) / (p.sum() + g.sum() + eps))


# =======================================================================
# Hausdorff Distance + ASSD (approx.)
# =======================================================================
def hausdorff_approx(p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """
    Approximate the Hausdorff Distance (HD) and Average Symmetric Surface Distance (ASSD)
    using distance transforms on binary masks p, g ∈ {0,1}.

    Note:
        - This approximation is widely used when computing exact HD is too expensive.
        - Returns large sentinel values if segmentation is completely missing.

    Returns:
        (HD, ASSD) as floats.
    """
    p = (p > 0).astype("uint8")
    g = (g > 0).astype("uint8")

    # Case 1: both masks are empty → perfect match (no lesion)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0, 0.0

    # Case 2: one mask empty, one non-empty → failure case
    if p.sum() == 0 or g.sum() == 0:
        return 99.0, 99.0

    # Compute Euclidean Distance Transform (EDT)
    dp = edt(1 - p)
    dg = edt(1 - g)

    # Hausdorff: maximum symmetric surface distance
    hd = max(dp[g.astype(bool)].max(), dg[p.astype(bool)].max())

    # ASSD: mean symmetric surface distance
    s1 = dp[g.astype(bool)].mean()
    s2 = dg[p.astype(bool)].mean()

    return float(hd), float((s1 + s2) / 2.0)


# =======================================================================
# Main Evaluation Function
# =======================================================================
@torch.no_grad()
def main(ckpt: str, splits_csv: str, img_size: int) -> None:
    """
    Evaluate segmentation quality (DSC, HD, ASSD) on the test split.

    Args:
        ckpt: path to model checkpoint (.pt file).
        splits_csv: CSV file containing dataset split.
        img_size: image resolution for inference.
    """

    # -------------------------------------------------------------------
    # Dataset loader (test split only)
    # -------------------------------------------------------------------
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    # -------------------------------------------------------------------
    # Model setup
    # -------------------------------------------------------------------
    model = MACLOSiamNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Lists to store metrics over batches
    dscs: list[float] = []
    hds: list[float] = []
    assds: list[float] = []

    # -------------------------------------------------------------------
    # Inference loop
    # -------------------------------------------------------------------
    for batch in dl:
        # Model forward pass: segmentation logits only
        seg_logits, _, _ = model(batch["mri"], batch["ct"])  # [B,1,H,W]

        # Convert logits → probabilities
        seg_probs = torch.sigmoid(seg_logits)

        # Move to CPU and convert to NumPy
        p = seg_probs.squeeze(1).cpu().numpy()  # [B,H,W]
        g = batch["seg"].squeeze(1).cpu().numpy()  # [B,H,W]

        # -------------------------------------------------------------------
        # Compute metrics per sample
        # -------------------------------------------------------------------
        for i in range(p.shape[0]):
            # Threshold probability map at 0.5 → binary mask
            pred_mask = (p[i] > 0.5).astype("float32")
            gt_mask = g[i].astype("float32")

            # Compute metrics
            d = dice(pred_mask, gt_mask)
            hd, asd = hausdorff_approx(pred_mask, gt_mask)

            dscs.append(d)
            hds.append(hd)
            assds.append(asd)

    # -------------------------------------------------------------------
    # Save results to CSV
    # -------------------------------------------------------------------
    os.makedirs("results/tables", exist_ok=True)
    out_path = "results/tables/seg_metrics.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DSC", "HD", "ASSD"])
        w.writerow([np.mean(dscs), np.mean(hds), np.mean(assds)])

    print(f"[OK] segmentation metrics saved → {out_path}")


# =======================================================================
# CLI Entry Point
# =======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Segmentation Metrics Evaluation Script")
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    ap.add_argument("--splits", required=True, help="Path to splits CSV file")
    ap.add_argument("--img_size", type=int, default=256, help="Input resolution")
    args = ap.parse_args()

    main(args.ckpt, args.splits, args.img_size)
