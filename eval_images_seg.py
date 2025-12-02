import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt as edt

from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet


def dice(p: np.ndarray, g: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Dice coefficient between two binary masks p, g in {0,1}.
    """
    p = (p > 0).astype("float32")
    g = (g > 0).astype("float32")
    inter = (p * g).sum()
    return float((2.0 * inter + eps) / (p.sum() + g.sum() + eps))


def hausdorff_approx(p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """
    Approximate Hausdorff distance (HD) and Average Symmetric Surface Distance (ASSD)
    using distance transforms on binary masks p, g in {0,1}.
    """
    p = (p > 0).astype("uint8")
    g = (g > 0).astype("uint8")

    if p.sum() == 0 and g.sum() == 0:
        return 0.0, 0.0

    if p.sum() == 0 or g.sum() == 0:
        return 99.0, 99.0

    dp = edt(1 - p)
    dg = edt(1 - g)

    hd = max(dp[g.astype(bool)].max(), dg[p.astype(bool)].max())
    s1 = dp[g.astype(bool)].mean()
    s2 = dg[p.astype(bool)].mean()

    return float(hd), float((s1 + s2) / 2.0)


@torch.no_grad()
def main(ckpt: str, splits_csv: str, img_size: int) -> None:
    """
    Evaluate segmentation quality (DSC, HD, ASSD) on the test split.
    """

    # --- Dataset & loader ---
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    # --- Model ---
    model = MACLOSiamNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dscs: list[float] = []
    hds: list[float] = []
    assds: list[float] = []

    # --- Inference loop ---
    for batch in dl:
        seg_logits, _, _ = model(batch["mri"], batch["ct"])  # [B, 1, H, W] (logits or probs)

        # إذا الموديل يرجع logits، نستخدم sigmoid:
        seg_probs = torch.sigmoid(seg_logits)

        # تحويل إلى NumPy على CPU
        p = seg_probs.squeeze(1).cpu().numpy()  # [B, H, W]
        g = batch["seg"].squeeze(1).cpu().numpy()  # [B, H, W]

        for i in range(p.shape[0]):
            pred_mask = (p[i] > 0.5).astype("float32")
            gt_mask = g[i].astype("float32")

            d = dice(pred_mask, gt_mask)
            hd, asd = hausdorff_approx(pred_mask, gt_mask)

            dscs.append(d)
            hds.append(hd)
            assds.append(asd)

    # --- Aggregate and save ---
    os.makedirs("results/tables", exist_ok=True)
    out_path = "results/tables/seg_metrics.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DSC", "HD", "ASSD"])
        w.writerow([np.mean(dscs), np.mean(hds), np.mean(assds)])

    print(f"[OK] segmentation metrics saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    main(args.ckpt, args.splits, args.img_size)
