# train_maclo_siamnet_cv.py
import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model import MACLOSiamNet
from maclo_utils import compute_task_losses, maclo_unified_loss


# ========= Reproducibility =========
def set_seed(seed: int = 2025) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========= Example Dataset class (patient-level) =========
class AISDataset(Dataset):
    """
    Example AIS dataset for patient-level cross-validation.

    data_index: list of dicts, each dict e.g.:
        {
          "patient_id": int or str,
          "modality": "MRI" or "CT",
          "image_path": "...",      # path to preprocessed slice/volume
          "seg_label_path": "...",
          "cls_label": int,         # classification label
          "age_label": float        # lesion-age label (hours)
        }

    NOTE:
        - This class currently uses random tensors as placeholders for images
          and metadata to illustrate the training pipeline.
        - In the actual implementation used for the paper, this should be
          replaced with real I/O loading MRI/CT volumes and clinical metadata.
    """

    def __init__(self, data_index, transform=None):
        self.data_index = data_index
        self.transform = transform

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        item = self.data_index[idx]

        # TODO: Replace the following placeholders with real image/metadata loading.
        # For example:
        #   - load MRI and CT slices from item["image_path"]
        #   - load segmentation mask from item["seg_label_path"]
        #   - build metadata vector from clinical variables

        x_mri = torch.randn(1, 256, 256)  # [C,H,W] MRI placeholder
        x_ct = torch.randn(1, 256, 256)   # [C,H,W] CT placeholder
        meta = torch.randn(16)            # example metadata embedding

        seg = torch.randint(0, 2, (1, 256, 256)).float()  # binary mask placeholder
        cls = torch.tensor(item["cls_label"], dtype=torch.long)
        age = torch.tensor(item["age_label"], dtype=torch.float32)

        sample = {
            "mri": x_mri,
            "ct": x_ct,
            "meta": meta,
            "seg": seg,
            "cls": cls,
            "age": age,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# ========= Utility: build patient-level folds =========
def build_patient_folds(patient_meta, n_splits: int = 5, seed: int = 2025):
    """
    Build patient-level stratified folds based on combined clinical factors.

    Args:
        patient_meta: list of dicts per patient, e.g.:
            {
              "patient_id": ...,
              "stroke_subtype": int,
              "lesion_size_cat": int,
              "lesion_age_cat": int
            }
        n_splits: number of folds for cross-validation.
        seed: random seed for StratifiedKFold.

    We create a combined stratification label from (stroke_subtype, size_cat, age_cat)
    to preserve the joint distribution across folds.
    """
    ids = [p["patient_id"] for p in patient_meta]
    y_strat = []
    for p in patient_meta:
        # combine subtype + lesion size category + lesion-age category
        code = (p["stroke_subtype"], p["lesion_size_cat"], p["lesion_age_cat"])
        y_strat.append(code)

    # map tuple codes to integer labels for StratifiedKFold
    unique_codes = {c: i for i, c in enumerate(sorted(set(y_strat)))}
    y = np.array([unique_codes[c] for c in y_strat])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids, y)):
        folds.append(
            {
                "train_val_ids": [ids[i] for i in train_val_idx],
                "test_ids": [ids[i] for i in test_idx],
            }
        )
    return folds


# ========= Training & evaluation for one fold =========
def train_one_fold(
    fold_idx,
    model,
    train_index,
    val_index,
    test_index,
    all_data_index,
    device: str = "cuda",
    batch_size: int = 4,
    lr: float = 1e-4,
    num_epochs: int = 50,
):
    """
    Train and evaluate MACLO-SiamNet on a single cross-validation fold.

    Args:
        fold_idx: index of the current fold.
        model: instance of MACLOSiamNet.
        train_index: list of patient_ids assigned to training.
        val_index: list of patient_ids assigned to validation.
        test_index: list of patient_ids assigned to test.
        all_data_index: list of all samples, each with a "patient_id" key.
        device: "cuda" or "cpu".
        batch_size: training batch size.
        lr: learning rate for Adam optimizer.
        num_epochs: number of training epochs for this fold.

    Returns:
        model: the trained model (with best validation snapshot loaded).
    """
    # Filter data_index by patient ids for this fold
    train_data = [d for d in all_data_index if d["patient_id"] in train_index]
    val_data = [d for d in all_data_index if d["patient_id"] in val_index]
    test_data = [d for d in all_data_index if d["patient_id"] in test_index]

    train_ds = AISDataset(train_data)
    val_ds = AISDataset(val_data)
    test_ds = AISDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        # ===== Training phase =====
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            mri = batch["mri"].to(device)
            ct = batch["ct"].to(device)
            meta = batch["meta"].to(device)

            seg_gt = batch["seg"].to(device)
            cls_gt = batch["cls"].to(device)
            age_gt = batch["age"].to(device)

            optimizer.zero_grad()

            # We assume MACLOSiamNet implements:
            #   forward(mri, ct, meta=None, return_z=False)
            # and returns:
            #   if return_z=True -> (Z_shared, seg_logits, cls_logits, age_pred)
            Z_shared, seg_p, cls_p, age_p = model(
                mri, ct, meta, return_z=True
            )

            outputs = {
                "seg": seg_p,
                "cls": cls_p,
                "age": age_p,
            }
            targets = {
                "seg": seg_gt,
                "cls": cls_gt,
                "age": age_gt,
            }

            # Per-task losses (Dice+BCE, CE, MAE) as defined in maclo_utils.py
            L_seg, L_cls, L_age = compute_task_losses(outputs, targets)

            # MACLO unified multi-task loss (gradient-based task re-weighting)
            L_maclo, lam = maclo_unified_loss(
                Z_shared,
                (L_seg, L_cls, L_age),
            )

            L_maclo.backward()
            optimizer.step()

            running_loss += L_maclo.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        # ===== Validation phase =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mri = batch["mri"].to(device)
                ct = batch["ct"].to(device)
                meta = batch["meta"].to(device)

                seg_gt = batch["seg"].to(device)
                cls_gt = batch["cls"].to(device)
                age_gt = batch["age"].to(device)

                Z_shared, seg_p, cls_p, age_p = model(
                    mri, ct, meta, return_z=True
                )

                outputs = {
                    "seg": seg_p,
                    "cls": cls_p,
                    "age": age_p,
                }
                targets = {
                    "seg": seg_gt,
                    "cls": cls_gt,
                    "age": age_gt,
                }

                L_seg, L_cls, L_age = compute_task_losses(outputs, targets)
                L_fold, _ = maclo_unified_loss(
                    Z_shared,
                    (L_seg, L_cls, L_age),
                )
                val_loss += L_fold.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(
            f"[Fold {fold_idx}] Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save best validation snapshot (early stopping)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    # Load the best validation snapshot for final testing
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== Test evaluation (placeholder) =====
    # Here you can implement detailed evaluation on test_loader:
    #   - segmentation: DSC, HD, ASSD
    #   - classification: accuracy, F1, AUC
    #   - age regression: MAE, MSE, R²
    model.eval()
    # TODO: implement computation of final test metrics

    return model


# ========= Main CV driver =========
def main():
    set_seed(2025)

    # 1) Load patient-level metadata (for stratification)
    with open("patients_meta.json", "r") as f:
        patient_meta = json.load(f)

    folds = build_patient_folds(patient_meta, n_splits=5, seed=2025)

    # 2) Load full index of slices/volumes linked to patient_id
    with open("data_index.json", "r") as f:
        all_data_index = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_fold_metrics = []
    for fold_idx, fold in enumerate(folds):
        train_val_ids = fold["train_val_ids"]
        test_ids = fold["test_ids"]

        # Split train_val into train and val (e.g. 80/20 split)
        n_train_val = len(train_val_ids)
        n_val = max(1, int(0.2 * n_train_val))
        random.shuffle(train_val_ids)
        val_ids = train_val_ids[:n_val]
        train_ids = train_val_ids[n_val:]

        # Instantiate a fresh model for this fold
        model = MACLOSiamNet()  # defined in model.py

        model = train_one_fold(
            fold_idx=fold_idx,
            model=model,
            train_index=train_ids,
            val_index=val_ids,
            test_index=test_ids,
            all_data_index=all_data_index,
            device=device,
            batch_size=4,
            lr=1e-4,
            num_epochs=50,
        )

        # TODO: collect test metrics for this fold and append to all_fold_metrics

    # TODO: aggregate metrics across folds and save as JSON/CSV for reporting
    # e.g., mean ± std of DSC, ACC, MAE, etc.


if __name__ == "__main__":
    main()
