# train_maclo_siamnet_cv.py
import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from maclo_utils import compute_task_losses, maclo_unified_loss
# from model import MACLOSiamNet  # يفترض أن يكون لديك ملف model.py فيه تعريف الشبكة


# ========= reproducibility =========
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========= Example Dataset class (patient-level) =========
class AISDataset(Dataset):
    """
    Example dataset:
    - data_index: list of dicts, each dict:
        {
          'patient_id': int or str,
          'modality': 'MRI' or 'CT',
          'image_path': '...',  # path to preprocessed slice/volume
          'seg_label_path': '...',
          'cls_label': int,
          'age_label': float
        }
    - transform: preprocessing / augmentation callable
    """
    def __init__(self, data_index, transform=None):
        self.data_index = data_index
        self.transform = transform

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        item = self.data_index[idx]
        # TODO: load MRI/CT slices, metadata, and labels.
        # For the reviewers, you can leave high-level placeholders,
        # but in real code you implement actual I/O.
        # Example:
        x_mri = torch.randn(1, 256, 256)  # placeholder
        x_ct  = torch.randn(1, 256, 256)  # placeholder
        meta  = torch.randn(16)           # e.g., metadata embedding

        seg = torch.randint(0, 2, (1, 256, 256)).float()
        cls = torch.tensor(item['cls_label'], dtype=torch.long)
        age = torch.tensor(item['age_label'], dtype=torch.float32)

        sample = {
            'mri': x_mri,
            'ct': x_ct,
            'meta': meta,
            'seg': seg,
            'cls': cls,
            'age': age,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# ========= Utility: build patient-level folds =========
def build_patient_folds(patient_meta, n_splits=5, seed=2025):
    """
    patient_meta: list of dicts per patient:
      {
        'patient_id': ...,
        'stroke_subtype': int,
        'lesion_size_cat': int,
        'lesion_age_cat': int
      }

    We build a stratification label y_strat from combined clinical factors.
    """
    ids = [p['patient_id'] for p in patient_meta]
    y_strat = []
    for p in patient_meta:
        # encode combined label, e.g. subtype + size_cat + age_cat
        code = (p['stroke_subtype'], p['lesion_size_cat'], p['lesion_age_cat'])
        y_strat.append(code)

    # Convert tuples to integers for StratifiedKFold (simple mapping):
    unique_codes = {c: i for i, c in enumerate(sorted(set(y_strat)))}
    y = np.array([unique_codes[c] for c in y_strat])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids, y)):
        # داخل train_val نعمل split إلى train/val لاحقاً
        folds.append({
            'train_val_ids': [ids[i] for i in train_val_idx],
            'test_ids':      [ids[i] for i in test_idx],
        })
    return folds


# ========= Training & evaluation for one fold =========
def train_one_fold(
    fold_idx,
    model,
    train_index,
    val_index,
    test_index,
    all_data_index,
    device='cuda',
    batch_size=4,
    lr=1e-4,
    num_epochs=50,
):

    # build data lists
    train_data = [d for d in all_data_index if d['patient_id'] in train_index]
    val_data   = [d for d in all_data_index if d['patient_id'] in val_index]
    test_data  = [d for d in all_data_index if d['patient_id'] in test_index]

    train_ds = AISDataset(train_data)
    val_ds   = AISDataset(val_data)
    test_ds  = AISDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            mri  = batch['mri'].to(device)
            ct   = batch['ct'].to(device)
            meta = batch['meta'].to(device)

            seg_gt = batch['seg'].to(device)
            cls_gt = batch['cls'].to(device)
            age_gt = batch['age'].to(device)

            optimizer.zero_grad()

            outputs, Z_shared = model(mri, ct, meta)  # assume model returns dict + shared feature

            targets = {'seg': seg_gt, 'cls': cls_gt, 'age': age_gt}
            L_seg, L_cls, L_age = compute_task_losses(outputs, targets)

            L_maclo, lam = maclo_unified_loss(model, Z_shared, (L_seg, L_cls, L_age))
            L_maclo.backward()
            optimizer.step()

            running_loss += L_maclo.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mri  = batch['mri'].to(device)
                ct   = batch['ct'].to(device)
                meta = batch['meta'].to(device)

                seg_gt = batch['seg'].to(device)
                cls_gt = batch['cls'].to(device)
                age_gt = batch['age'].to(device)

                outputs, Z_shared = model(mri, ct, meta)
                targets = {'seg': seg_gt, 'cls': cls_gt, 'age': age_gt}
                L_seg, L_cls, L_age = compute_task_losses(outputs, targets)
                L_maclo, _ = maclo_unified_loss(model, Z_shared, (L_seg, L_cls, L_age))
                val_loss += L_maclo.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"[Fold {fold_idx}] Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # early stopping snapshot
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation ( DSC, HD, ACC, MAE, إلخ)
    # 
    model.eval()
    # TODO: implement detailed metric computation

    return model  # fold


# ========= Main CV driver =========
def main():
    set_seed(2025)

    # 1) Load patient-level meta 
    with open("patients_meta.json", "r") as f:
        patient_meta = json.load(f)

    folds = build_patient_folds(patient_meta, n_splits=5, seed=2025)

    # 2) Load full index of slices/volumes linked to patient_id
    with open("data_index.json", "r") as f:
        all_data_index = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_fold_metrics = []
    for fold_idx, fold in enumerate(folds):
        train_val_ids = fold['train_val_ids']
        test_ids      = fold['test_ids']

        # split train_val into train/val (
        n_train_val = len(train_val_ids)
        n_val = max(1, int(0.2 * n_train_val))
        random.shuffle(train_val_ids)
        val_ids   = train_val_ids[:n_val]
        train_ids = train_val_ids[n_val:]

        # instantiate model fold
        model = MACLOSiamNet()  #  model.py

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

        # TODO: collect metrics for each fold and append to all_fold_metrics

    # ـ metrics
    # print / save results as JSON or CSV


if __name__ == "__main__":
    main()
