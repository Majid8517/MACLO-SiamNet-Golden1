from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import yaml

from maclo_v2.dataset import _load_gray
from maclo_v2.metadata import ClinicalMetadataEncoder
from maclo_v3 import MACLOClassifierV3


class V3Dataset(Dataset):
    def __init__(self, csv_path: str, split: str, image_size: int, metadata_encoder, augment: bool):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split") == split:
                    self.rows.append(row)
        self.image_size = image_size
        self.metadata_encoder = metadata_encoder
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _augment(self, x):
        if not self.augment:
            return x
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(()) < 0.5:
            scale = 0.9 + 0.2 * torch.rand(())
            shift = -0.05 + 0.10 * torch.rand(())
            x = (x * scale + shift).clamp(0, 1)
        if torch.rand(()) < 0.25:
            x = (x + torch.randn_like(x) * 0.015).clamp(0, 1)
        return x

    def __getitem__(self, i):
        row = self.rows[i]
        path = row.get("image_png_path", "")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Missing PNG for patient {row.get('patient_id')}: {path}")
        image = self._augment(_load_gray(path, self.image_size))
        clinical = self.metadata_encoder.transform_row(row)
        label = torch.tensor(int(row["cls_label"]), dtype=torch.long)
        return image, clinical, label, row["patient_id"]


@torch.no_grad()
def val_auc(model, loader, device, use_clinical: bool):
    model.eval()
    y, p = [], []
    for image, clinical, label, _ in loader:
        image = image.to(device)
        clinical = clinical.to(device)
        out = model(image, clinical if use_clinical else None)
        prob = torch.softmax(out["logits"], dim=1)[:, 1]
        y.extend(label.numpy().astype(int).tolist())
        p.extend(prob.cpu().numpy().tolist())
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ClinicalMetadataEncoder.fit_csv(args.csv, "train")
    if encoder.dim != int(cfg["clinical_dim"]):
        raise ValueError(f"clinical_dim mismatch: config={cfg['clinical_dim']}, encoder={encoder.dim}")

    train_ds = V3Dataset(args.csv, "train", int(cfg["image_size"]), encoder, augment=True)
    val_ds = V3Dataset(args.csv, "val", int(cfg["image_size"]), encoder, augment=False)
    train_dl = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0)

    model = MACLOClassifierV3(
        clinical_dim=int(cfg["clinical_dim"]),
        num_classes=2,
        channels=tuple(cfg["model"]["channels"]),
        depths=tuple(cfg["model"]["depths"]),
        fusion_mode=cfg["model"]["fusion_mode"],
        dropout=float(cfg["model"]["dropout"]),
        max_drop_path=float(cfg["model"]["max_drop_path"]),
        ccrf_strength=float(cfg["model"].get("ccrf_strength", 0.35)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg["training"]["max_epochs"]), eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))

    use_clinical = cfg["model"]["fusion_mode"] != "image_only"
    best_auc, bad = -1.0, 0
    ckpt = Path(args.checkpoint); ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        total = 0.0
        for image, clinical, label, _ in train_dl:
            image, clinical, label = image.to(device), clinical.to(device), label.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(image, clinical if use_clinical else None)
            loss = criterion(out["logits"], label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
        scheduler.step()

        auc = val_auc(model, val_dl, device, use_clinical)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch={epoch+1:03d} train_loss_sum={total:.4f} val_auc={auc:.4f} lr={lr:.7f}")

        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "metadata_encoder": {"means": encoder.means, "stds": encoder.stds, "dim": encoder.dim},
                "best_val_auc": best_auc,
            }, ckpt)
        else:
            bad += 1
            if bad >= int(cfg["training"]["early_stopping_patience"]):
                print("Early stopping.")
                break

    print(f"Best checkpoint: {ckpt} | val_auc={best_auc:.6f}")


if __name__ == "__main__":
    main()
