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
    Evaluate stroke classification on the test split and compute:
    - AUC
    - Accuracy
    - F1-score
    """

    # --- Dataset & loader ---
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    # --- Model ---
    model = MACLOSiamNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []

    # --- Inference loop ---
    for batch in dl:
        # model returns (segmentation, classification, age)
        _, cls_logits, _ = model(batch["mri"], batch["ct"])

        # true labels (B,)
        y_true += batch["cls"].cpu().numpy().tolist()

        # predicted probability for class 1 (B,)
        probs = torch.softmax(cls_logits, dim=1)[:, 1]
        y_prob += probs.cpu().numpy().tolist()

    # --- Metrics ---
    y_prob_arr = np.array(y_prob)
    y_pred_arr = (y_prob_arr > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_arr)
    f1 = f1_score(y_true, y_pred_arr)

    try:
        auc = roc_auc_score(y_true, y_prob_arr)
    except ValueError:
        # e.g., only one class present in y_true
        auc = 0.5

    # --- Save results ---
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cls_metrics.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AUC", "Accuracy", "F1"])
        writer.writerow([auc, acc, f1])

    print(f"[OK] classification metrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    main(args.ckpt, args.splits, args.img_size)
