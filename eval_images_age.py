import argparse, os, csv, numpy as np, torch
from torch.utils.data import DataLoader
from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main(ckpt, splits_csv, img_size):
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = MACLOSiamNet(); model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.eval()
    y_true=[]; y_pred=[]
    for b in dl:
        with torch.no_grad():
            _,_,age = model(b["mri"], b["ct"])
        y_true += b["age"].numpy().tolist()
        y_pred += age.squeeze(1).numpy().tolist()
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred)**0.5
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/age_metrics.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["R2","MAE","RMSE"]); w.writerow([R2, MAE, RMSE])
    print("[OK] age metrics -> results/tables/age_metrics.csv")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    a=ap.parse_args(); main(a.ckpt, a.splits, a.img_size)
