from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from maclo_v2.metadata import ClinicalMetadataEncoder
from maclo_v2.metrics import binary_classification_metrics


METRICS = [
    "accuracy", "sensitivity", "specificity", "precision",
    "f1", "auc", "pr_auc", "mcc", "kappa"
]


def read_split(csv_path: Path, split: str):
    df = pd.read_csv(csv_path)
    part = df[df["split"] == split].copy()
    if part.empty:
        raise ValueError(f"No rows found for split={split!r} in {csv_path}")
    return part


def encode_rows(df: pd.DataFrame, encoder: ClinicalMetadataEncoder):
    xs = []
    for row in df.to_dict(orient="records"):
        xs.append(encoder.transform_row(row).numpy())
    return np.stack(xs).astype(np.float32)


def build_model(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(name)


def save_fold(out_dir: Path, ids, y_true, y_prob, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "y_true", "y_prob", "y_pred"])
        for pid, y, p in zip(ids, y_true, y_prob):
            w.writerow([pid, int(y), float(p), int(p >= 0.5)])


def aggregate_model(results_root: Path, model_name: str):
    fold_rows = []
    oof_rows = []

    for fold in range(5):
        fold_dir = results_root / model_name / f"fold_{fold}"

        with open(fold_dir / "metrics.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
        obj["fold"] = fold
        fold_rows.append(obj)

        pred = pd.read_csv(fold_dir / "predictions.csv")
        pred["fold"] = fold
        oof_rows.append(pred)

    summary = {}
    for key in METRICS:
        vals = np.asarray([r[key] for r in fold_rows], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
        }

    model_dir = results_root / model_name
    pd.DataFrame(
        [
            {
                "metric": key,
                "mean": summary[key]["mean"],
                "std": summary[key]["std"],
                "mean±std": f'{summary[key]["mean"]:.4f} ± {summary[key]["std"]:.4f}',
            }
            for key in METRICS
        ]
    ).to_csv(model_dir / "cv_summary.csv", index=False)

    with open(model_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"folds": fold_rows, "summary": summary}, f, indent=2)

    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(model_dir / "oof_predictions.csv", index=False)

    pooled = binary_classification_metrics(oof["y_true"].values, oof["y_prob"].values)
    with open(model_dir / "oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pooled, f, indent=2)

    return summary, pooled


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Leakage-audited classical clinical baselines on the same patient-level "
            "five-fold splits used by the deep models."
        )
    )
    ap.add_argument("--fold-dir", default="generated_index")
    ap.add_argument("--results-root", default="results_classical")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest"],
        choices=["logistic_regression", "random_forest"],
    )
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fold_dir = Path(args.fold_dir)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")

        for fold in range(5):
            csv_path = fold_dir / f"fold_{fold}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)

            # Fit all preprocessing statistics on TRAIN ONLY.
            encoder = ClinicalMetadataEncoder.fit_csv(str(csv_path), split="train")

            train_df = read_split(csv_path, "train")
            test_df = read_split(csv_path, "test")

            X_train = encode_rows(train_df, encoder)
            X_test = encode_rows(test_df, encoder)

            y_train = train_df["cls_label"].astype(int).to_numpy()
            y_test = test_df["cls_label"].astype(int).to_numpy()
            ids = test_df["patient_id"].astype(str).tolist()

            model = build_model(model_name, args.seed + fold)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = binary_classification_metrics(y_test, y_prob, threshold=0.5)
            metrics["fold"] = fold
            metrics["n_train"] = int(len(train_df))
            metrics["n_test"] = int(len(test_df))
            metrics["clinical_dim"] = int(encoder.dim)
            metrics["d_dimer_used"] = False

            out_dir = results_root / model_name / f"fold_{fold}"
            save_fold(out_dir, ids, y_test, y_prob, metrics)

            print(
                f"Fold {fold}: "
                f"AUC={metrics['auc']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"MCC={metrics['mcc']:.4f}"
            )

        summary, pooled = aggregate_model(results_root, model_name)

        print("\n5-fold mean ± std")
        for key in METRICS:
            m = summary[key]["mean"]
            s = summary[key]["std"]
            print(f"{key:12s}: {m:.4f} ± {s:.4f}")

        print("\nPooled OOF metrics")
        print(json.dumps(pooled, indent=2))

    print(
        "\nNOTE: Clinical features are encoded with the same leakage-audited "
        "ClinicalMetadataEncoder used by the deep metadata branch; d_dimer is excluded."
    )


if __name__ == "__main__":
    main()
