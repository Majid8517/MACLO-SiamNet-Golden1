from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


CONFIRMATORY_SEEDS = [2026, 31415, 27182, 16180, 42424]


def main():
    ap = argparse.ArgumentParser(
        description="Create frozen repeated 5-fold patient-level confirmatory splits."
    )
    ap.add_argument("--master-csv", default="generated_index/master_dataset.csv")
    ap.add_argument("--out-dir", default="confirmatory_folds")
    args = ap.parse_args()

    df = pd.read_csv(args.master_csv)
    required = {"patient_id", "cls_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient_id values found in master dataset.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "frozen_repeated_stratified_5fold",
        "n_patients": int(len(df)),
        "class_counts": {
            str(k): int(v)
            for k, v in df["cls_label"].value_counts().sort_index().items()
        },
        "seeds": CONFIRMATORY_SEEDS,
        "n_repeats": len(CONFIRMATORY_SEEDS),
        "n_splits": 5,
        "validation_rule": "val_fold=(test_fold+1)%5 within each repeat",
        "notes": [
            "These folds are distinct from the development folds generated with seed 2025.",
            "Architecture and hyperparameters are frozen before these confirmatory runs.",
            "No test-fold tuning or threshold optimization is permitted.",
        ],
        "repeats": {},
    }

    for repeat_idx, seed in enumerate(CONFIRMATORY_SEEDS):
        rep_dir = out_root / f"repeat_{repeat_idx:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        work = df.copy()
        work["confirmatory_fold"] = -1

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, test_idx) in enumerate(skf.split(work, work["cls_label"])):
            work.loc[test_idx, "confirmatory_fold"] = fold

        # Every patient must appear in exactly one test fold per repeat.
        if (work["confirmatory_fold"] < 0).any():
            raise RuntimeError(f"Unassigned patient in repeat {repeat_idx}")

        repeat_info = {
            "seed": seed,
            "fold_class_counts": {},
        }

        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5
            fold_df = work.copy()
            fold_df["split"] = "train"
            fold_df.loc[fold_df["confirmatory_fold"] == val_fold, "split"] = "val"
            fold_df.loc[fold_df["confirmatory_fold"] == test_fold, "split"] = "test"

            # Sanity checks.
            if set(fold_df["split"].unique()) != {"train", "val", "test"}:
                raise RuntimeError("Invalid split assignment.")
            if fold_df.loc[fold_df["split"] == "test", "patient_id"].duplicated().any():
                raise RuntimeError("Duplicate test patient.")

            out_path = rep_dir / f"fold_{test_fold}.csv"
            fold_df.to_csv(out_path, index=False)

            counts = (
                fold_df[fold_df["split"] == "test"]["cls_label"]
                .value_counts()
                .sort_index()
            )
            repeat_info["fold_class_counts"][str(test_fold)] = {
                str(k): int(v) for k, v in counts.items()
            }

        manifest["repeats"][str(repeat_idx)] = repeat_info

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    print(f"Saved confirmatory folds to: {out_root}")


if __name__ == "__main__":
    main()
