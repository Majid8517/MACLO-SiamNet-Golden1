import argparse
import os

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet
from .utils import set_seed


def main(cfg_path: str, paths_path: str, img_size: int) -> None:
    # --- Load configs ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(paths_path, "r") as f:
        paths = yaml.safe_load(f)

    set_seed(cfg["SEED"])

    # --- Device ---
    device_str = cfg.get("DEVICE", "cpu")
    if device_str != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # --- Model & optimizer ---
    model = MACLOSiamNet(img_size=img_size).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["TRAIN"]["LR"])
    )

    # --- Loss functions ---
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    l1  = nn.L1Loss()

    # --- Datasets & loaders ---
    splits_csv = paths["DATA"]["SPLIT_CSV"]

    train_ds = ImageFolderAIS(splits_csv, split="train", img_size=img_size)
    val_ds   = ImageFolderAIS(splits_csv, split="val",   img_size=img_size)

    tr_loader = DataLoader(
        train_ds,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
    )

    ckpt_root = paths.get("CHECKPOINT_ROOT", "./checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    best = float("inf")

    # --- Training loop ---
    for epoch in range(cfg["TRAIN"]["EPOCHS"]):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            mri = batch["mri"].to(device)    # [B,1,H,W]
            ct  = batch["ct"].to(device)     # [B,1,H,W]
            seg_t = batch["seg"].to(device)  # [B,1,H,W]
            cls_t = batch["cls"].to(device)  # [B]
            age_t = batch["age"].to(device)  # [B]

            # forward: seg, cls, age
            seg_p, cls_p, age_p = model(mri, ct)

            losses = {
                "seg": bce(seg_p, seg_t),
                "cls": ce(cls_p, cls_t),
                "age": l1(age_p.squeeze(1), age_t),
            }
            loss = sum(losses.values())

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in losses.items()})

        # --- Validation: هنا استخدمت فقط loss العمر كمثال ---
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mri = batch["mri"].to(device)
                ct  = batch["ct"].to(device)
                age_t = batch["age"].to(device)

                _, _, age_p = model(mri, ct)
                vloss += l1(age_p.squeeze(1), age_t).item()

        if vloss < best:
            best = vloss
            ckpt_path = os.path.join(ckpt_root, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] New best val loss: {best:.4f}, saved to {ckpt_path}")

    print("[OK] training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()
    main(args.cfg, args.paths, args.img_size)
