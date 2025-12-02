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
    Evaluate lesion-age estimation on the test split and compute:
        - R² (coefficient of determination)
        - MAE (mean absolute error)
        - RMSE (root mean squared error)

    Args:
        ckpt:       Path to the model checkpoint (.pt or .pth).
        splits_csv: Path to the CSV file defining the dataset splits
                    (must contain a 'test' split).
        img_size:   Image size used to resize MRI/CT images during loading.
    """

    # ------------------------------------------------------------------
    # 1) Dataset and DataLoader (test split only)
    # ------------------------------------------------------------------
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    # ------------------------------------------------------------------
    # 2) Load model and checkpoint
    # ------------------------------------------------------------------
    model = MACLOSiamNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Containers for ground-truth and predicted lesion ages
    y_true: list[float] = []
    y_pred: list[float] = []

    # ------------------------------------------------------------------
    # 3) Inference loop
    # ------------------------------------------------------------------
    for batch in dl:
        # Forward pass:
        #   MACLOSiamNet returns (segmentation_logits, classification_logits, age_pred)
        _, _, age_pred = model(batch["mri"], batch["ct"])

        # Move tensors to CPU and convert to plain Python lists
        y_true += batch["age"].cpu().numpy().tolist()
        y_pred += age_pred.squeeze(1).cpu().numpy().tolist()

    # ------------------------------------------------------------------
    # 4) Compute regression metrics
    # ------------------------------------------------------------------
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred) ** 0.5

    # ------------------------------------------------------------------
    # 5) Save metrics to CSV
    # ------------------------------------------------------------------
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "age_metrics.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R2", "MAE", "RMSE"])
        writer.writerow([R2, MAE, RMSE])

    print(f"[OK] lesion-age metrics saved to {out_path}")


# ======================================================================
# Command-line interface
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate lesion-age estimation metrics on the test split."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to model checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--splits",
        required=True,
        help="Path to dataset splits CSV file.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (default: 256).",
    )
    args = parser.parse_args()

    main(args.ckpt, args.splits, args.img_size)
