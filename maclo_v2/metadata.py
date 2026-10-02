from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List
import csv, json, math
import numpy as np
import torch

MISSING = {"", "NA", "N/A", "NaN", "nan", "None", None}

NUMERIC_FIELDS = ["age", "avg_glucose_level", "cholesterol", "bmi"]
BINARY_FIELDS = ["hypertension", "heart_disease"]
CATEGORICAL_FIELDS = {
    "gender": ["Female", "Male", "Unknown"],
    "d_dimer": ["Negative", "Positive", "Unknown"],
    "marital_status": ["No", "Yes", "Unknown"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked", "Unknown"],
    "residence_type": ["Rural", "Urban", "Unknown"],
    "smoking_status": ["never", "formerly", "smokes", "Unknown"],
}


def _to_float(value):
    if value in MISSING:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ClinicalMetadataEncoder:
    means: Dict[str, float]
    stds: Dict[str, float]

    @property
    def dim(self) -> int:
        return 2 * len(NUMERIC_FIELDS) + len(BINARY_FIELDS) + sum(len(v) for v in CATEGORICAL_FIELDS.values())

    @classmethod
    def fit_csv(cls, csv_path: str, split: str = "train"):
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == split:
                    rows.append(row)
        if not rows:
            raise ValueError(f"No rows found for split={split!r}")
        return cls.fit_rows(rows)

    @classmethod
    def fit_rows(cls, rows: Iterable[dict]):
        rows = list(rows)
        means, stds = {}, {}
        for field in NUMERIC_FIELDS:
            vals = [_to_float(r.get(field)) for r in rows]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if not vals:
                means[field], stds[field] = 0.0, 1.0
            else:
                means[field] = float(np.mean(vals))
                std = float(np.std(vals))
                stds[field] = std if std > 1e-8 else 1.0
        return cls(means, stds)

    def transform_row(self, row: dict) -> torch.Tensor:
        out: List[float] = []

        for field in NUMERIC_FIELDS:
            value = _to_float(row.get(field))
            if value is None or not math.isfinite(value):
                out.extend([0.0, 1.0])
            else:
                out.extend([(value - self.means[field]) / self.stds[field], 0.0])

        for field in BINARY_FIELDS:
            value = _to_float(row.get(field))
            out.append(0.0 if value is None else float(value))

        for field, vocab in CATEGORICAL_FIELDS.items():
            raw = str(row.get(field, "Unknown")).strip()
            if field == "d_dimer" and raw == "Posotive":
                raw = "Positive"
            if raw not in vocab:
                raw = "Unknown"
            out.extend([1.0 if raw == item else 0.0 for item in vocab])

        return torch.tensor(out, dtype=torch.float32)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"means": self.means, "stds": self.stds, "dim": self.dim}, handle, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        return cls(obj["means"], obj["stds"])
