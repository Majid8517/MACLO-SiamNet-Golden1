import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet


@torch.no_grad()
def main(ckpt: str, splits_csv: str, img_size: int) -> None:
    """
    Evaluate lesion-age estimation on the test split and compute metrics:
    R2, MAE, RMSE.

    Args:
        ckpt:      path to model checkpoint (.pt or .pth)
        splits_csv: CSV containing all samples with test rows
        img_size:  resizing size for MRI/CT images
    """

    # --- Dataset ---
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    # --- Model ---
    model = MACLOSiamNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    y_true: list[float] = []
    y_pred: list[float] = []

    # --- Inference loop ---
    for batch in dl:
        seg, cls, age = model(batch["mri"], batch["ct"])  # forward returns 3 heads

        # Convert to CPU numpy
        y_true += batch["age"].cpu().numpy().tolist()
        y_pred += age.squeeze(1).cpu().numpy().tolist()

    # --- Metrics ---
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred) ** 0.5

    # --- Save results ---
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "age_metrics.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R2", "MAE", "RMSE"])
        w.writerow([R2, MAE, RMSE])

    print(f"[OK] Age metrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    main(args.ckpt, args.splits, args.img_size)
