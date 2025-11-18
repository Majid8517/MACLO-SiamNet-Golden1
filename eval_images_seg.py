import argparse, os, csv, numpy as np, torch
from torch.utils.data import DataLoader
from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet
from scipy.ndimage import distance_transform_edt as edt

def dice(p, g, eps=1e-6):
    p = (p>0).astype("float32"); g = (g>0).astype("float32")
    inter = (p*g).sum()
    return (2*inter + eps) / (p.sum()+g.sum()+eps)

def hausdorff_approx(p, g):
    p = (p>0).astype("uint8"); g = (g>0).astype("uint8")
    if p.sum()==0 and g.sum()==0: return 0.0, 0.0
    if p.sum()==0 or g.sum()==0:  return 99.0, 99.0
    dp = edt(1-p); dg = edt(1-g)
    hd = max(dp[g.astype(bool)].max(), dg[p.astype(bool)].max())
    s1 = dp[g.astype(bool)].mean(); s2 = dg[p.astype(bool)].mean()
    return float(hd), float((s1+s2)/2.0)

def main(ckpt, splits_csv, img_size):
    ds = ImageFolderAIS(splits_csv, split="test", img_size=img_size)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    model = MACLOSiamNet(); model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.eval()
    dscs, hds, assds = [], [], []
    for b in dl:
        with torch.no_grad():
            seg,_,_ = model(b["mri"], b["ct"])
        p = seg.numpy(); g = b["seg"].numpy()
        for i in range(p.shape[0]):
            d = dice(p[i,0], g[i,0])
            hd, asd = hausdorff_approx(p[i,0], g[i,0])
            dscs.append(d); hds.append(hd); assds.append(asd)
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/seg_metrics.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["DSC","HD","ASSD"]); w.writerow([np.mean(dscs), np.mean(hds), np.mean(assds)])
    print("[OK] seg metrics -> results/tables/seg_metrics.csv")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    a=ap.parse_args(); main(a.ckpt, a.splits, a.img_size)
