"""
dataset_images.py

Dataset utilities for MACLO-SiamNet:
- MRI / CT image loading (PNG / JPG / DICOM)
- Optional binary lesion masks
- Classification labels (stroke subtype)
- Lesion-age regression targets (in hours)
- Optional JSON metadata per sample

The dataset is driven by a CSV file with at least the following columns:

    split        : "train" / "val" / "test" (or similar)
    mri_path     : path to MRI image (PNG/JPG/DCM)
    ct_path      : path to CT image (PNG/JPG/DCM)
    mask_path    : path to binary mask image (may be empty)
    cls_label    : integer class label (may be empty -> default 0)
    age_hours    : lesion age in hours (float, may be empty -> default 0.0)
    meta_path    : path to JSON metadata file (may be empty)

Example row (CSV):

    split,mri_path,ct_path,mask_path,cls_label,age_hours,meta_path
    train,data/mri_001.dcm,data/ct_001.png,data/mask_001.png,1,5.0,data/meta_001.json
"""

from __future__ import annotations

import os
import csv
import json
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import pydicom


def _load_img(path: str, img_size: int) -> np.ndarray:
    """
    Load an MRI/CT image from PNG/JPG/DICOM, normalize to [0,1],
    and resize to (img_size, img_size).

    Returns:
        numpy array of shape [1, H, W] with float32 values in [0, 1].
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dcm":
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype("float32")
        # NOTE: optionally apply rescale slope/intercept if present
        # slope = getattr(dcm, "RescaleSlope", 1.0)
        # inter = getattr(dcm, "RescaleIntercept", 0.0)
        # arr = arr * slope + inter
    else:
        arr = np.array(Image.open(path).convert("L"), dtype="float32")

    # Min-max normalize per-slice
    arr_min = arr.min()
    arr_max = arr.max()
    arr = (arr - arr_min) / (arr_max - arr_min + 1e-6)

    # Resize to target resolution
    pil_img = Image.fromarray((arr * 255).astype("uint8"))
    pil_img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.array(pil_img, dtype="float32") / 255.0

    # Add channel dimension: [1, H, W]
    return arr[None, ...]


def _load_mask(path: str, img_size: int) -> np.ndarray:
    """
    Load a binary segmentation mask from an image file, resize with NEAREST,
    and return [1, H, W] float32 in {0,1}.
    """
    arr = np.array(Image.open(path).convert("L"), dtype="float32") / 255.0
    pil_img = Image.fromarray((arr * 255).astype("uint8"))
    pil_img = pil_img.resize((img_size, img_size), resample=Image.NEAREST)
    arr = (np.array(pil_img) > 127).astype("float32")
    return arr[1 - 1 : 2 - 1] if arr.ndim == 3 else arr[None, ...]  # ensure [1, H, W]


class ImageFolderAIS(Dataset):
    """
    Generic AIS image dataset for MACLO-SiamNet.

    Expects a CSV file with one row per sample and columns described
    in the module docstring.

    Args:
        splits_csv: path to the CSV file containing all samples.
        split:      split name to select (e.g. "train", "val", "test").
        img_size:   output H=W resolution for images and masks.
    """

    def __init__(self, splits_csv: str, split: str, img_size: int = 256) -> None:
        super().__init__()
        self.items = []

        with open(splits_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "") == split:
                    self.items.append(row)

        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.items[idx]

        mri_path = row.get("mri_path", "")
        ct_path = row.get("ct_path", "")
        mask_path = row.get("mask_path", "")
        cls_str = row.get("cls_label", "")
        age_str = row.get("age_hours", "")
        meta_path = row.get("meta_path", "")

        # --- Images (MRI / CT) ---
        mri_np = _load_img(mri_path, self.img_size)
        ct_np = _load_img(ct_path, self.img_size)

        # --- Segmentation mask (optional) ---
        if mask_path and os.path.exists(mask_path):
            seg_np = _load_mask(mask_path, self.img_size)
        else:
            seg_np = np.zeros_like(mri_np, dtype="float32")

        # --- Classification label (optional) ---
        cls = int(cls_str) if cls_str not in ("", None) else 0

        # --- Lesion age in hours (optional) ---
        age = float(age_str) if age_str not in ("", None) else 0.0

        # --- Metadata (optional JSON) ---
        meta: Dict[str, Any] = {}
        if meta_path and os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        sample = {
            "mri": torch.from_numpy(mri_np).float(),          # [1, H, W]
            "ct": torch.from_numpy(ct_np).float(),            # [1, H, W]
            "seg": torch.from_numpy(seg_np).float(),          # [1, H, W]
            "cls": torch.tensor(cls, dtype=torch.long),       # []
            "age": torch.tensor(age, dtype=torch.float32),    # []
            "meta": meta,                                     # raw dict (optional)
        }

        return sample
