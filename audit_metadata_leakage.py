from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_EXCLUDE = {
    "cls_label", "label", "target", "patient_id", "image_path", "filepath",
    "file_path", "path", "split", "fold", "source", "image_id"
}


def safe_auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return np.nan
    auc = roc_auc_score(y, score)
    # Direction-invariant screening score: a strong inverse relationship is also a shortcut risk.
    return float(max(auc, 1.0 - auc))


def numeric_oof_auc(x: pd.Series, y: pd.Series, seed: int, folds: int):
    X = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce")})
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return safe_auc(y, p)


def categorical_oof_auc(x: pd.Series, y: pd.Series, seed: int, folds: int):
    X = pd.DataFrame({"x": x.astype("object")})
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["x"])
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return safe_auc(y, p)


def missingness_auc(x: pd.Series, y: pd.Series):
    m = x.isna().astype(int)
    if m.nunique() < 2:
        return np.nan
    return safe_auc(y, m)


def cramers_v(x: pd.Series, y: pd.Series):
    tab = pd.crosstab(x.fillna("MISSING").astype(str), y)
    if tab.empty:
        return np.nan
    obs = tab.to_numpy(dtype=float)
    n = obs.sum()
    if n <= 1:
        return np.nan
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    expected = row @ col / n
    mask = expected > 0
    chi2 = (((obs - expected) ** 2) / np.where(mask, expected, 1))[mask].sum()
    r, k = obs.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def main():
    ap = argparse.ArgumentParser(
        description="Full leakage audit for structured clinical variables."
    )
    ap.add_argument("--csv", required=True, help="Master/index CSV containing cls_label.")
    ap.add_argument("--out-dir", default="audit_results")
    ap.add_argument("--label-col", default="cls_label")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional columns to exclude from predictive screening.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    y = pd.to_numeric(df[args.label_col], errors="raise").astype(int)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise ValueError("Leakage audit currently expects a binary 0/1 label.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded = DEFAULT_EXCLUDE | {args.label_col} | set(args.exclude)
    candidate_cols = [c for c in df.columns if c not in excluded]

    rows = []
    crosstabs = {}

    for col in candidate_cols:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        missing_rate = float(s.isna().mean())

        numeric = pd.to_numeric(s, errors="coerce")
        numeric_fraction = float(numeric.notna().mean())

        # Treat genuinely numeric columns as numeric; otherwise use categorical OOF encoding.
        is_numeric = numeric_fraction >= 0.95 and nunique > 2

        if is_numeric:
            feature_type = "numeric"
            auc = numeric_oof_auc(s, y, args.seed, args.folds)
            association = np.nan
        else:
            feature_type = "categorical"
            auc = categorical_oof_auc(s, y, args.seed, args.folds)
            association = cramers_v(s, y)
            tab = pd.crosstab(s.fillna("MISSING"), y)
            crosstabs[col] = {
                str(idx): {str(k): int(v) for k, v in row.items()}
                for idx, row in tab.to_dict(orient="index").items()
            }

        miss_auc = missingness_auc(s, y)

        # Deterministic category-label mapping flag.
        deterministic = False
        if not is_numeric:
            tab = pd.crosstab(s.fillna("MISSING"), y)
            deterministic = bool(
                tab.shape[0] > 0 and ((tab > 0).sum(axis=1) <= 1).all()
            )

        # Heuristic audit flag only; not a formal statistical conclusion.
        if deterministic or (not np.isnan(auc) and auc >= 0.95):
            risk = "CRITICAL"
        elif not np.isnan(auc) and auc >= 0.85:
            risk = "HIGH"
        elif not np.isnan(auc) and auc >= 0.75:
            risk = "MODERATE"
        else:
            risk = "LOW"

        rows.append({
            "feature": col,
            "type": feature_type,
            "n_unique": nunique,
            "missing_rate": missing_rate,
            "oof_univariate_auc_direction_invariant": auc,
            "missingness_auc_direction_invariant": miss_auc,
            "cramers_v_if_categorical": association,
            "deterministic_category_label_mapping": deterministic,
            "leakage_risk_flag": risk,
        })

    audit = pd.DataFrame(rows).sort_values(
        ["leakage_risk_flag", "oof_univariate_auc_direction_invariant"],
        ascending=[True, False],
    )

    # Re-order risk labels by explicit severity for readable output.
    risk_order = pd.CategoricalDtype(
        categories=["CRITICAL", "HIGH", "MODERATE", "LOW"], ordered=True
    )
    audit["risk_order"] = audit["leakage_risk_flag"].astype(risk_order)
    audit = audit.sort_values(
        ["risk_order", "oof_univariate_auc_direction_invariant"],
        ascending=[True, False],
    ).drop(columns=["risk_order"])

    csv_path = out_dir / "clinical_feature_leakage_audit.csv"
    json_path = out_dir / "clinical_feature_leakage_audit.json"
    audit.to_csv(csv_path, index=False)

    payload = {
        "n_patients": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "label_column": args.label_col,
        "seed": args.seed,
        "cv_folds": args.folds,
        "screening_note": (
            "Univariate AUCs are out-of-fold and direction-invariant. Risk flags are "
            "screening heuristics, not proof of causal leakage. Features with high AUC "
            "must be reviewed for temporal availability, label derivation, and clinical plausibility."
        ),
        "features": audit.replace({np.nan: None}).to_dict(orient="records"),
        "categorical_crosstabs": crosstabs,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== FULL CLINICAL FEATURE LEAKAGE AUDIT ===")
    print(f"Patients: {len(df)} | Class counts: {dict(y.value_counts().sort_index())}")
    print(audit.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    critical = audit[audit["leakage_risk_flag"] == "CRITICAL"]["feature"].tolist()
    high = audit[audit["leakage_risk_flag"] == "HIGH"]["feature"].tolist()
    print(f"\nCritical-review features: {critical}")
    print(f"High-review features: {high}")


if __name__ == "__main__":
    main()
