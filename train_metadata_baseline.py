from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from maclo_v2.metadata import ClinicalMetadataEncoder
from maclo_v2.metrics import binary_classification_metrics


class MetaDataset(Dataset):
    def __init__(self, csv_path, split, encoder):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split") == split:
                    self.rows.append(row)
        self.encoder = encoder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return self.encoder.transform_row(row), torch.tensor(int(row["cls_label"])), row["patient_id"]


class MetaMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def infer(model, loader, device):
    ids, y, p = [], [], []
    model.eval()
    for x, yy, pid in loader:
        prob = torch.softmax(model(x.to(device)), dim=1)[:,1].cpu().numpy()
        ids.extend(list(pid))
        y.extend(yy.numpy().astype(int).tolist())
        p.extend(prob.tolist())
    return ids, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ClinicalMetadataEncoder.fit_csv(args.csv, "train")
    train_ds = MetaDataset(args.csv, "train", encoder)
    val_ds = MetaDataset(args.csv, "val", encoder)
    test_ds = MetaDataset(args.csv, "test", encoder)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    test_dl = DataLoader(test_ds, batch_size=64)

    model = MetaMLP(encoder.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y, _ in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()

        _, vy, vp = infer(model, val_dl, device)
        val_auc = roc_auc_score(vy, vp)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    ids, y, p = infer(model, test_dl, device)
    metrics = binary_classification_metrics(y, p)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out/"metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2)
    with open(out/"predictions.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["patient_id","y_true","y_prob","y_pred"])
        for pid,yy,pp in zip(ids,y,p):
            w.writerow([pid,yy,pp,int(pp>=0.5)])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
