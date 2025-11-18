import argparse, os, csv, numpy as np, torch
from torch.utils.data import DataLoader
from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def main(ckpt, splits_csv, img_size):
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = MACLOSiamNet(); model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.eval()
    y_true=[]; y_prob=[]
    for b in dl:
        with torch.no_grad():
            _,cls,_ = model(b["mri"], b["ct"])
        y_true += b["cls"].numpy().tolist()
        y_prob += torch.softmax(cls,1)[:,1].numpy().tolist()
    y_pred = (np.array(y_prob)>0.5).astype(int)
    A = accuracy_score(y_true, y_pred)
    F = f1_score(y_true, y_pred)
    try: U = roc_auc_score(y_true, y_prob)
    except: U = 0.5
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/cls_metrics.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["AUC","Accuracy","F1"]); w.writerow([U, A, F])
    print("[OK] cls metrics -> results/tables/cls_metrics.csv")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    a=ap.parse_args(); main(a.ckpt, a.splits, a.img_size)
