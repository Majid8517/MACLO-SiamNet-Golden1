from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import roc_auc_score

from maclo_v2 import MACLOSiamNetV2, HeterogeneousStrokeDataset, compute_task_losses, MACLOController
from maclo_v2.metadata import ClinicalMetadataEncoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_nested(batch, device):
    inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
    availability = batch["availability"].to(device)
    metadata = batch["metadata"].to(device)
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    masks = {k: v.to(device) for k, v in batch["task_mask"].items()}
    return inputs, availability, metadata, targets, masks


def active_losses(losses, masks):
    return {
        task: loss
        for task, loss in losses.items()
        if task in masks and masks[task].any().item()
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, batches = 0.0, 0
    y_true, y_prob = [], []

    for batch in loader:
        inputs, availability, metadata, targets, masks = move_nested(batch, device)
        outputs = model(
            inputs, availability,
            metadata if metadata.shape[-1] > 0 else None
        )
        losses = active_losses(compute_task_losses(outputs, targets, masks), masks)
        if losses:
            total_loss += float(torch.stack(list(losses.values())).sum().cpu())
            batches += 1

        if "cls" in outputs:
            keep = masks["cls"].bool()
            if keep.any():
                probs = torch.softmax(outputs["cls"], dim=1)[:, 1]
                y_true.extend(targets["cls"][keep].detach().cpu().numpy().astype(int).tolist())
                y_prob.extend(probs[keep].detach().cpu().numpy().tolist())

    mean_loss = total_loss / max(batches, 1)
    auc = float("nan")
    if len(set(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    return {"val_loss": mean_loss, "val_auc": auc}


def main(csv_path: str, config_path: str, checkpoint_override: str | None = None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = list(cfg["modalities"])

    metadata_encoder = None
    if cfg.get("metadata", {}).get("enabled", False):
        metadata_encoder = ClinicalMetadataEncoder.fit_csv(csv_path, split="train")
        if metadata_encoder.dim != int(cfg["meta_dim"]):
            raise ValueError(
                f"meta_dim={cfg['meta_dim']} but encoder produces {metadata_encoder.dim} features."
            )

    train_ds = HeterogeneousStrokeDataset(
        csv_path, modalities, "train", int(cfg["image_size"]), metadata_encoder
    )
    val_ds = HeterogeneousStrokeDataset(
        csv_path, modalities, "val", int(cfg["image_size"]), metadata_encoder
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True, num_workers=int(cfg["training"]["num_workers"])
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False, num_workers=int(cfg["training"]["num_workers"])
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

    selection_metric = cfg.get("evaluation", {}).get("selection_metric", "val_loss")
    maximize = selection_metric in {"val_auc"}
    best_score = -float("inf") if maximize else float("inf")
    bad_epochs = 0

    checkpoint = Path(
        checkpoint_override
        or cfg["training"].get("checkpoint", "checkpoints_v2/best.pt")
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        running = 0.0
        cosine_log, neg_log = [], []

        for batch in train_loader:
            inputs, availability, metadata, targets, masks = move_nested(batch, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                inputs, availability,
                metadata if metadata.shape[-1] > 0 else None
            )
            losses = active_losses(compute_task_losses(outputs, targets, masks), masks)
            if not losses:
                continue

            shared_params, unified_parts, maclo_stats = maclo.compute_unified_gradients(
                losses, model.shared_parameters()
            )

            total = torch.stack(list(losses.values())).sum()
            total.backward()
            maclo.apply_unified_gradients(shared_params, unified_parts)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(total.detach().cpu())

            cosine_log.append(maclo_stats["cosine_matrix"].cpu())
            neg_log.append(float(maclo_stats["negative_pair_fraction"].cpu()))

        val = validate(model, val_loader, device)
        score = val.get(selection_metric, val["val_loss"])
        if np.isnan(score):
            score = -float("inf") if maximize else float("inf")

        mean_neg = float(np.mean(neg_log)) if neg_log else float("nan")
        print(
            f"epoch={epoch+1:03d} train_sum={running:.4f} "
            f"val_loss={val['val_loss']:.4f} val_auc={val['val_auc']:.4f} "
            f"neg_grad_fraction={mean_neg:.4f}"
        )

        improved = score > best_score if maximize else score < best_score
        if improved:
            best_score = score
            bad_epochs = 0
            state = {
                "model": model.state_dict(),
                "config": cfg,
                "selection_metric": selection_metric,
                "best_score": best_score,
            }
            if metadata_encoder is not None:
                state["metadata_encoder"] = {
                    "means": metadata_encoder.means,
                    "stds": metadata_encoder.stds,
                    "dim": metadata_encoder.dim,
                }
            torch.save(state, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"]["early_stopping_patience"]):
                print("Early stopping on validation data only.")
                break

    print(f"Best checkpoint: {checkpoint} | {selection_metric}={best_score:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.csv, args.config, args.checkpoint)
