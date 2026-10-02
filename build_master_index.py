from __future__ import annotations
import argparse, json, os, re, zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def normalize_patient_id(value: str):
    text = str(value).strip()
    group = re.match(r"\s*([12])", text)
    numbers = re.findall(r"\d+", text)
    if group is None or len(numbers) < 2:
        raise ValueError(f"Cannot normalize Patient ID: {value!r}")
    prefix = group.group(1)
    index = int(numbers[-1])
    return f"{prefix}({index})", prefix, index


def scan_directory(root: Path):
    result = defaultdict(dict)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        m = re.fullmatch(r"([12]) \((\d+)\)\.(jpg|JPG|png)", path.name)
        if not m:
            continue
        pid = f"{m.group(1)}({int(m.group(2))})"
        result[pid][m.group(3).lower()] = str(path.resolve())
    return result


def scan_zip(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    result = defaultdict(dict)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = os.path.basename(member)
            m = re.fullmatch(r"([12]) \((\d+)\)\.(jpg|JPG|png)", name)
            if not m:
                continue
            pid = f"{m.group(1)}({int(m.group(2))})"
            target = extract_to / pid.replace("(", "_").replace(")", "") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            result[pid][m.group(3).lower()] = str(target.resolve())
    return result


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--images-root")
    source.add_argument("--images-zip")
    ap.add_argument("--extract-to")
    ap.add_argument("--output-dir", default="generated_index")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.images_root:
        image_map = scan_directory(Path(args.images_root))
    else:
        if not args.extract_to:
            raise ValueError("--extract-to is required with --images-zip")
        image_map = scan_zip(Path(args.images_zip), Path(args.extract_to))

    frame = pd.read_excel(args.excel)
    expected = {
        "Patient ID", "Gender", "Age", "Hypertension", "Avg_Glucose_level",
        "Cholesterol levels", "BMI", "D-dimer test", "Heart_Disease",
        "Marital Status", "Work Type", "Residence_type", "Smoking Status",
        "Brain Stroke",
    }
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Excel is missing columns: {sorted(missing)}")

    records, corrections = [], []
    for _, row in frame.iterrows():
        canonical, _, _ = normalize_patient_id(row["Patient ID"])
        if clean_text(row["Patient ID"]) != canonical:
            corrections.append({"source": clean_text(row["Patient ID"]), "canonical": canonical})

        images = image_map.get(canonical, {})
        d_dimer = clean_text(row["D-dimer test"]).replace("Posotive", "Positive")

        records.append({
            "patient_id": canonical,
            "source_patient_id": clean_text(row["Patient ID"]),
            "dataset": "stroke_normal_paper2",
            "image_png_path": images.get("png", ""),
            "image_jpg_path": images.get("jpg", ""),
            "cls_label": int(row["Brain Stroke"]),
            "gender": clean_text(row["Gender"]),
            "age": row["Age"] if not pd.isna(row["Age"]) else "",
            "hypertension": row["Hypertension"] if not pd.isna(row["Hypertension"]) else "",
            "avg_glucose_level": row["Avg_Glucose_level"] if not pd.isna(row["Avg_Glucose_level"]) else "",
            "cholesterol": row["Cholesterol levels"] if not pd.isna(row["Cholesterol levels"]) else "",
            "bmi": row["BMI"] if not pd.isna(row["BMI"]) else "",
            "d_dimer": d_dimer,
            "heart_disease": row["Heart_Disease"] if not pd.isna(row["Heart_Disease"]) else "",
            "marital_status": clean_text(row["Marital Status"]),
            "work_type": clean_text(row["Work Type"]),
            "residence_type": clean_text(row["Residence_type"]),
            "smoking_status": clean_text(row["Smoking Status"]),
            "age_hours": "",
            "mask_path": "",
            "modality_label": "unverified",
        })

    out = pd.DataFrame(records)
    if out["patient_id"].duplicated().any():
        dup = out.loc[out["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Duplicate normalized patient IDs: {dup[:20]}")

    no_image = out.loc[
        (out["image_png_path"] == "") & (out["image_jpg_path"] == ""),
        "patient_id"
    ].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    out["outer_fold"] = -1
    for fold, (_, test_idx) in enumerate(skf.split(out, out["cls_label"])):
        out.loc[test_idx, "outer_fold"] = fold

    master_path = output_dir / "master_dataset.csv"
    out.to_csv(master_path, index=False)

    for test_fold in range(5):
        val_fold = (test_fold + 1) % 5
        fold = out.copy()
        fold["split"] = "train"
        fold.loc[fold["outer_fold"] == val_fold, "split"] = "val"
        fold.loc[fold["outer_fold"] == test_fold, "split"] = "test"
        fold.to_csv(output_dir / f"fold_{test_fold}.csv", index=False)

    image_ids = set(image_map.keys())
    meta_ids = set(out["patient_id"])
    report = {
        "metadata_rows": int(len(out)),
        "class_counts": {str(k): int(v) for k, v in out["cls_label"].value_counts().sort_index().items()},
        "matched_png": int((out["image_png_path"] != "").sum()),
        "matched_jpg": int((out["image_jpg_path"] != "").sum()),
        "matched_both": int(((out["image_png_path"] != "") & (out["image_jpg_path"] != "")).sum()),
        "metadata_without_any_image": no_image,
        "image_ids_without_metadata": sorted(image_ids - meta_ids),
        "id_corrections": corrections,
        "fold_class_counts": {
            str(fold): {
                str(label): int(count)
                for label, count in out[out["outer_fold"] == fold]["cls_label"].value_counts().sort_index().items()
            }
            for fold in range(5)
        },
        "modality_note": (
            "File extension is not used as an MRI/CT label. image_png and image_jpg "
            "remain generic image sources until modality provenance is verified."
        ),
    }
    with open(output_dir / "linkage_audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {master_path}")


if __name__ == "__main__":
    main()
