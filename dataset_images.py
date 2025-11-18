import os, csv, json, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import pydicom

def _load_img(path, img_size):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        d = pydicom.dcmread(path)
        arr = d.pixel_array.astype("float32")
    else:
        arr = np.array(Image.open(path).convert("L"), dtype="float32")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    from PIL import Image as PILImage
    arr_img = PILImage.fromarray((arr*255).astype("uint8"))
    arr_img = arr_img.resize((img_size, img_size), resample=PILImage.BILINEAR)
    arr = np.array(arr_img).astype("float32")/255.0
    return arr[None, ...]

def _load_mask(path, img_size):
    arr = np.array(Image.open(path).convert("L"), dtype="float32")/255.0
    from PIL import Image as PILImage
    arr_img = PILImage.fromarray((arr*255).astype("uint8"))
    arr_img = arr_img.resize((img_size, img_size), resample=PILImage.NEAREST)
    arr = (np.array(arr_img)>127).astype("float32")
    return arr[None, ...]

class ImageFolderAIS(Dataset):
    def __init__(self, splits_csv, split, img_size=256):
        self.items = []
        import csv as _csv
        with open(splits_csv) as f:
            r = _csv.DictReader(f)
            for row in r:
                if row["split"] == split:
                    self.items.append(row)
        self.img_size = img_size

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        mri = _load_img(it["mri_path"], self.img_size)
        ct  = _load_img(it["ct_path"], self.img_size)
        seg = _load_mask(it["mask_path"], self.img_size) if it["mask_path"] else np.zeros_like(mri)
        cls = int(it["cls_label"]) if it["cls_label"] != "" else 0
        age = float(it["age_hours"]) if it["age_hours"] != "" else 0.0

        meta = {}
        if it["meta_path"] and os.path.exists(it["meta_path"]) and os.path.getsize(it["meta_path"])>0:
            with open(it["meta_path"]) as f:
                try: meta = json.load(f)
                except: meta = {}

        return {"mri": torch.from_numpy(mri).float(),
                "ct":  torch.from_numpy(ct).float(),
                "seg": torch.from_numpy(seg).float(),
                "cls": torch.tensor(cls, dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "meta": meta}
