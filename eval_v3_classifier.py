from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from maclo_v2.metadata import ClinicalMetadataEncoder
from maclo_v2.metrics import binary_classification_metrics
from maclo_v3 import MACLOClassifierV3
from train_v3_classifier import V3Dataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    m = state["metadata_encoder"]
    encoder = ClinicalMetadataEncoder(m["means"], m["stds"])

    ds = V3Dataset(args.csv, "test", int(cfg["image_size"]), encoder, augment=False)
    dl = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    model = MACLOClassifierV3(
        clinical_dim=int(cfg["clinical_dim"]),
        num_classes=2,
        channels=tuple(cfg["model"]["channels"]),
        depths=tuple(cfg["model"]["depths"]),
        fusion_mode=cfg["model"]["fusion_mode"],
        dropout=float(cfg["model"]["dropout"]),
        max_drop_path=float(cfg["model"]["max_drop_path"]),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    use_clinical = cfg["model"]["fusion_mode"] != "image_only"
    ids, y, p, gates = [], [], [], []

    for image, clinical, label, patient_id in dl:
        image, clinical = image.to(device), clinical.to(device)
        out = model(image, clinical if use_clinical else None)
        prob = torch.softmax(out["logits"], dim=1)[:, 1]
        ids.extend(list(patient_id))
        y.extend(label.numpy().astype(int).tolist())
        p.extend(prob.cpu().numpy().tolist())
        if out["gate_weights"] is not None:
            gates.extend(out["gate_weights"].cpu().numpy().tolist())

    metrics = binary_classification_metrics(y, p)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with open(out_dir/"predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["patient_id","y_true","y_prob","y_pred"]
        if gates:
            header += ["image_gate","clinical_gate"]
        w.writerow(header)
        for i,(pid,yy,pp) in enumerate(zip(ids,y,p)):
            row = [pid,yy,pp,int(pp>=0.5)]
            if gates:
                row += gates[i]
            w.writerow(row)

    if gates:
        g = np.asarray(gates)
        gate_summary = {
            "image_gate_mean": float(g[:,0].mean()),
            "image_gate_std": float(g[:,0].std(ddof=1)),
            "clinical_gate_mean": float(g[:,1].mean()),
            "clinical_gate_std": float(g[:,1].std(ddof=1)),
        }
        (out_dir/"gate_summary.json").write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
