from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from maclo_v2 import (
    MACLOSiamNetV2,
    HeterogeneousStrokeDataset,
    compute_task_losses,
    MACLOController,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_nested(batch, device):
    inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
    availability = batch["availability"].to(device)
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    masks = {k: v.to(device) for k, v in batch["task_mask"].items()}
    return inputs, availability, targets, masks


def active_losses(losses, masks):
    active = {}
    for task, loss in losses.items():
        if task in masks and masks[task].any().item():
            active[task] = loss
    return active


def main(csv_path: str, config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = list(cfg["modalities"])

    train_ds = HeterogeneousStrokeDataset(
        csv_path, modalities, "train", int(cfg["image_size"])
    )
    val_ds = HeterogeneousStrokeDataset(
        csv_path, modalities, "val", int(cfg["image_size"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )

    model = MACLOSiamNetV2(
        modality_names=modalities,
        channels=tuple(cfg["model"]["channels"]),
        meta_dim=int(cfg["meta_dim"]),
        num_classes=cfg["num_classes"],
        spam_keep_ratio=float(cfg["model"]["spam_keep_ratio"]),
        scct_heads=int(cfg["model"]["scct_heads"]),
        scct_depth=int(cfg["model"]["scct_depth"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    maclo = MACLOController(
        conflict_strength=float(cfg["maclo"]["conflict_strength"]),
        temperature=float(cfg["maclo"]["temperature"]),
    )

    best_val = float("inf")
    bad_epochs = 0
    checkpoint = Path("checkpoints_v2/best.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        running = 0.0

        for batch in train_loader:
            inputs, availability, targets, masks = move_nested(batch, device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, availability)
            losses = compute_task_losses(outputs, targets, masks)
            losses = active_losses(losses, masks)
            if not losses:
                continue

            # First obtain task gradients for the shared trunk while graph is intact.
            stats = maclo.task_gradients(losses, model.shared_parameters())

            # Ordinary backward supplies task-head/non-shared gradients.
            total = torch.stack(list(losses.values())).sum()
            total.backward()

            # Recompute and overwrite only the designated shared gradients
            # with the MACLO unified gradient.
            maclo_stats = maclo.overwrite_shared_gradients(
                losses, model.shared_parameters()
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(total.detach().cpu())

        model.eval()
        val_total, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, availability, targets, masks = move_nested(batch, device)
                outputs = model(inputs, availability)
                losses = compute_task_losses(outputs, targets, masks)
                losses = active_losses(losses, masks)
                if losses:
                    val_total += float(torch.stack(list(losses.values())).sum().cpu())
                    val_batches += 1

        mean_val = val_total / max(val_batches, 1)
        print(
            f"epoch={epoch+1:03d} train_sum={running:.4f} "
            f"val_sum={mean_val:.4f}"
        )

        if mean_val < best_val:
            best_val = mean_val
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "best_val": best_val,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"]["early_stopping_patience"]):
                print("Early stopping on validation data only.")
                break

    print(f"Best checkpoint: {checkpoint} | val={best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    main(args.csv, args.config)
