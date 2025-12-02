import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet


@torch.no_grad()
def main(ckpt: str, splits_csv: str, img_size: int) -> None:
    """
    Evaluate stroke classification performance on the test split.

    Metrics:
        - AUC (Area Under the ROC Curve)
        - Accuracy
        - F1-score (for the positive class)

    Args:
        ckpt:       Path to the trained model checkpoint (.pt file).
        splits_csv: Path to the CSV file defining the dataset splits.
        img_size:   Input image size used during training/inference.
    """

    # ------------------------------------------------------------------
    # 1) Dataset & DataLoader (test split only)
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

    # Containers to accumulate labels and probabilities across the test set
    y_true: list[int] = []
    y_prob: list[float] = []

    # ------------------------------------------------------------------
    # 3) Inference loop over the test set
    # ------------------------------------------------------------------
    for batch in dl:
        # Model forward pass:
        #   MACLOSiamNet returns (segmentation_logits, classification_logits, age_prediction)
        _, cls_logits, _ = model(batch["mri"], batch["ct"])

        # Ground-truth labels as a plain Python list
        y_true += batch["cls"].cpu().numpy().tolist()

        # Predicted probability for the positive class (class index 1)
        probs = torch.softmax(cls_logits, dim=1)[:, 1]
        y_prob += probs.cpu().numpy().tolist()

    # ------------------------------------------------------------------
    # 4) Compute classification metrics
    # ------------------------------------------------------------------
    y_prob_arr = np.array(y_prob)
    # Binary predictions using a fixed threshold at 0.5
    y_pred_arr = (y_prob_arr > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_arr)
    f1 = f1_score(y_true, y_pred_arr)

    # AUC can fail if only one class is present in y_true
    try:
        auc = roc_auc_score(y_true, y_prob_arr)
    except ValueError:
        # Fallback: non-informative classifier
        auc = 0.5

    # ------------------------------------------------------------------
    # 5) Save metrics to CSV
    # ------------------------------------------------------------------
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cls_metrics.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AUC", "Accuracy", "F1"])
        writer.writerow([auc, acc, f1])

    print(f"[OK] classification metrics saved to {out_path}")


# ======================================================================
# Command-line interface
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stroke classification metrics on the test split."
    )
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--splits", required=True, help="Path to dataset splits CSV.")
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (default: 256).",
    )
    args = parser.parse_args()

    main(args.ckpt, args.splits, args.img_size)
