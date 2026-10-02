from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from maclo_v2 import MACLOSiamNetV2, HeterogeneousStrokeDataset
from maclo_v2.metadata import ClinicalMetadataEncoder
from maclo_v2.metrics import binary_classification_metrics


def build_metadata_encoder(checkpoint):
    state = checkpoint.get("metadata_encoder")
    if state is None:
        return None
    return ClinicalMetadataEncoder(state["means"], state["stds"])


@torch.no_grad()
def evaluate(csv_path: str, config_path: str, checkpoint_path: str, split: str = "test"):
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    metadata_encoder = build_metadata_encoder(ckpt)
    ds = HeterogeneousStrokeDataset(
        csv_path,
        list(cfg["modalities"]),
        split,
        int(cfg["image_size"]),
        metadata_encoder,
    )
    dl = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    model = MACLOSiamNetV2(
        modality_names=list(cfg["modalities"]),
        channels=tuple(cfg["model"]["channels"]),
        meta_dim=int(cfg["meta_dim"]),
        num_classes=cfg["num_classes"],
        spam_keep_ratio=float(cfg["model"]["spam_keep_ratio"]),
        scct_heads=int(cfg["model"]["scct_heads"]),
        scct_depth=int(cfg["model"]["scct_depth"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids, y_true, y_prob = [], [], []

    for batch in dl:
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        availability = batch["availability"].to(device)
        metadata = batch["metadata"].to(device)
        targets = batch["targets"]["cls"].cpu().numpy()
        mask = batch["task_mask"]["cls"].cpu().numpy().astype(bool)

        outputs = model(
            inputs,
            availability,
            metadata if metadata.shape[-1] > 0 else None,
        )
        prob = torch.softmax(outputs["cls"], dim=1)[:, 1].cpu().numpy()

        batch_ids = list(batch["patient_id"])
        for pid, yy, pp, keep in zip(batch_ids, targets, prob, mask):
            if keep:
                ids.append(pid)
                y_true.append(int(yy))
                y_prob.append(float(pp))

    metrics = binary_classification_metrics(y_true, y_prob, threshold=0.5)
    return metrics, ids, y_true, y_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    metrics, ids, y_true, y_prob = evaluate(
        args.csv, args.config, args.checkpoint, args.split
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "y_true", "y_prob", "y_pred"])
        for pid, yy, pp in zip(ids, y_true, y_prob):
            w.writerow([pid, yy, pp, int(pp >= 0.5)])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
