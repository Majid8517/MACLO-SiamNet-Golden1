import argparse, yaml, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .dataset_images import ImageFolderAIS
from .model import MACLOSiamNet
from .utils import set_seed
from tqdm import tqdm

def main(cfg_path, paths_path, img_size):
    cfg = yaml.safe_load(open(cfg_path))
    paths = yaml.safe_load(open(paths_path))
    set_seed(cfg["SEED"])

    device = torch.device(cfg["DEVICE"])
    model = MACLOSiamNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["TRAIN"]["LR"]))

    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    l1  = nn.L1Loss()

    train_ds = ImageFolderAIS(paths["DATA"]["SPLIT_CSV"], split="train", img_size=img_size)
    val_ds   = ImageFolderAIS(paths["DATA"]["SPLIT_CSV"], split="val",   img_size=img_size)
    tr = DataLoader(train_ds, batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=True)
    va = DataLoader(val_ds,   batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=False)

    best = 1e9
    for epoch in range(cfg["TRAIN"]["EPOCHS"]):
        model.train(); pbar = tqdm(tr, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            mri, ct = batch["mri"].to(device), batch["ct"].to(device)
            seg_t, cls_t, age_t = batch["seg"].to(device), batch["cls"].to(device), batch["age"].to(device)

            seg_p, cls_p, age_p = model(mri, ct, metadata=batch.get("meta", None))
            losses = {
                "seg": bce(seg_p, seg_t),
                "cls": ce(cls_p, cls_t),
                "age": l1(age_p.squeeze(1), age_t)
            }
            loss = model.uw(losses)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({k: f"{v.item():.3f}" for k,v in losses.items()})

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for batch in va:
                mri, ct = batch["mri"].to(device), batch["ct"].to(device)
                seg_t, cls_t, age_t = batch["seg"].to(device), batch["cls"].to(device), batch["age"].to(device)
                seg_p, cls_p, age_p = model(mri, ct)
                vloss += l1(age_p.squeeze(1), age_t).item()
        if vloss < best:
            best = vloss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best.pt")
    print("[OK] training complete; saved checkpoints/best.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()
    main(args.cfg, args.paths, args.img_size)
