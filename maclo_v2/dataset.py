from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import csv

import numpy as np
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset

from .metadata import ClinicalMetadataEncoder

MISSING = {"", "NA", "N/A", "NaN", "nan", "None", None}


def _load_gray(path: str, image_size: int) -> torch.Tensor:
    suffix = Path(path).suffix.lower()
    if suffix == ".dcm":
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
    else:
        arr = np.asarray(Image.open(path).convert("F"), dtype=np.float32)

    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(float(hi - lo), 1e-6)
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None]).float()


def _load_mask(path: str, image_size: int) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    pil = Image.fromarray(arr).resize((image_size, image_size), Image.NEAREST)
    mask = (np.asarray(pil) > 0).astype(np.float32)
    return torch.from_numpy(mask[None]).float()


class HeterogeneousStrokeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        modalities: List[str],
        split: str,
        image_size: int = 256,
        metadata_encoder: Optional[ClinicalMetadataEncoder] = None,
    ):
        self.rows = []
        self.modalities = list(modalities)
        self.image_size = image_size
        self.metadata_encoder = metadata_encoder

        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == split:
                    self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        inputs: Dict[str, torch.Tensor] = {}
        availability = []

        for modality in self.modalities:
            path = row.get(f"{modality}_path", "")
            exists = path not in MISSING and Path(path).exists()
            availability.append(exists)
            inputs[modality] = (
                _load_gray(path, self.image_size)
                if exists
                else torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
            )

        if not any(availability):
            raise ValueError(
                f"Patient {row.get('patient_id', idx)} has no available configured modality."
            )

        mask_path = row.get("mask_path", "")
        has_seg = mask_path not in MISSING and Path(mask_path).exists()
        seg = (
            _load_mask(mask_path, self.image_size)
            if has_seg
            else torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        )

        age_raw = row.get("age_hours", "")
        has_age = age_raw not in MISSING
        age = float(age_raw) if has_age else 0.0

        cls_raw = row.get("cls_label", "")
        has_cls = cls_raw not in MISSING
        cls = int(cls_raw) if has_cls else 0

        metadata = (
            self.metadata_encoder.transform_row(row)
            if self.metadata_encoder is not None
            else torch.empty(0, dtype=torch.float32)
        )

        return {
            "patient_id": row.get("patient_id", str(idx)),
            "dataset": row.get("dataset", ""),
            "inputs": inputs,
            "availability": torch.tensor(availability, dtype=torch.bool),
            "metadata": metadata,
            "targets": {
                "seg": seg,
                "age": torch.tensor(age, dtype=torch.float32),
                "cls": torch.tensor(cls, dtype=torch.long),
            },
            "task_mask": {
                "seg": torch.tensor(has_seg, dtype=torch.bool),
                "age": torch.tensor(has_age, dtype=torch.bool),
                "cls": torch.tensor(has_cls, dtype=torch.bool),
            },
        }
