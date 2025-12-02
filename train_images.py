import argparse
import os

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet
from .maclo_utils import compute_task_losses, maclo_unified_loss
from .utils import set_seed


def main(cfg_path: str, paths_path: str, img_size: int) -> None:
    # -----------------------------
    # 1) Load configs
    # -----------------------------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(paths_path, "r") as f:
        paths = yaml.safe_load(f)

    set_seed(cfg["SEED"])

    # -----------------------------
    # 2) Device
    # -----------------------------
    device_str = cfg.get("DEVICE", "cpu")
    if device_str != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # -----------------------------
    # 3) Model & optimizer
    # -----------------------------
    model = MACLOSiamNet(img_size=img_size).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["TRAIN"]["LR"])
    )

    # -----------------------------
    # 4) Datasets & loaders
    # -----------------------------
    splits_csv = paths["DATA"]["SPLIT_CSV"]

    train_ds = ImageFolderAIS(
        splits_csv,
        split="train",
        img_size=img_size
    )
    val_ds = ImageFolderAIS(
        splits_csv,
        split="val",
        img_size=img_size
    )

    train_loader = DataLoader(
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

    best_val = float("inf")

    # -----------------------------
    # 5) Training loop with MACLO
    # -----------------------------
    num_epochs = cfg["TRAIN"]["EPOCHS"]
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{num_epochs}]")

        for batch in pbar:
            # ----- move batch to device -----
            mri = batch["mri"].to(device)       # [B,1,H,W]
            ct = batch["ct"].to(device)         # [B,1,H,W]
            seg_t = batch["seg"].to(device)     # [B,1,H,W]
            cls_t = batch["cls"].to(device)     # [B]
            age_t = batch["age"].to(device)     # [B]

            # ----- forward with Z for MACLO -----
            Z_shared, seg_p, cls_p, age_p = model(
                mri,
                ct,
                return_z=True
            )

            # ----- per-task losses (seg, cls, age) -----
            outputs = {
                "seg": seg_p,
                "cls": cls_p,
                "age": age_p,
            }
            targets = {
                "seg": seg_t,
                "cls": cls_t,
                "age": age_t,
            }

            L_seg, L_cls, L_age = compute_task_losses(outputs, targets)

            # ----- MACLO unified loss -----
            L_maclo, lambdas = maclo_unified_loss(
                Z_shared,
                (L_seg, L_cls, L_age)
            )

            optimizer.zero_grad()
            L_maclo.backward()
            optimizer.step()

            pbar.set_postfix({
                "L_seg": f"{L_seg.item():.3f}",
                "L_cls": f"{L_cls.item():.3f}",
                "L_age": f"{L_age.item():.3f}",
                "λ_s": f"{lambdas[0]:.2f}",
                "λ_c": f"{lambdas[1]:.2f}",
                "λ_a": f"{lambdas[2]:.2f}",
            })

        # -----------------------------
        # 6) Validation (simple: mean total loss)
        # -----------------------------
        model.eval()
        val_L_seg, val_L_cls, val_L_age = 0.0, 0.0, 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                mri = batch["mri"].to(device)
                ct = batch["ct"].to(device)
                seg_t = batch["seg"].to(device)
                cls_t = batch["cls"].to(device)
                age_t = batch["age"].to(device)

                seg_p, cls_p, age_p = model(mri, ct)

                outputs = {
                    "seg": seg_p,
                    "cls": cls_p,
                    "age": age_p,
                }
                targets = {
                    "seg": seg_t,
                    "cls": cls_t,
                    "age": age_t,
                }

                Ls, Lc, La = compute_task_losses(outputs, targets)
                val_L_seg += Ls.item()
                val_L_cls += Lc.item()
                val_L_age += La.item()
                n_val_batches += 1

        if n_val_batches > 0:
            val_L_seg /= n_val_batches
            val_L_cls /= n_val_batches
            val_L_age /= n_val_batches

        val_total = val_L_seg + val_L_cls + val_L_age
        print(
            f"[VAL] Epoch {epoch + 1}: "
            f"L_seg={val_L_seg:.4f}, "
            f"L_cls={val_L_cls:.4f}, "
            f"L_age={val_L_age:.4f}, "
            f"L_total={val_total:.4f}"
        )

        # -----------------------------
        # 7) Save best checkpoint
        # -----------------------------
        if val_total < best_val:
            best_val = val_total
            ckpt_path = os.path.join(ckpt_root, "best_maclo.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] New best val loss {best_val:.4f}, saved to {ckpt_path}")

    print("[OK] MACLO training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to main config YAML (seed, train, device...)")
    parser.add_argument("--paths", required=True, help="Path to paths YAML (DATA, CHECKPOINT_ROOT, etc.)")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    main(args.cfg, args.paths, args.img_size)
