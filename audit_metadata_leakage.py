from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="metadata_leakage_audit.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df["cls_label"].astype(int)

    report = {"categorical_crosstabs": {}, "numeric_univariate_auc": {}}

    for col in [
        "gender", "hypertension", "heart_disease", "marital_status",
        "work_type", "residence_type", "smoking_status", "d_dimer",
    ]:
        if col in df.columns:
            tab = pd.crosstab(df[col].fillna("MISSING"), y)
            report["categorical_crosstabs"][col] = {
                str(idx): {str(k): int(v) for k, v in row.items()}
                for idx, row in tab.to_dict(orient="index").items()
            }

    for col in ["age", "avg_glucose_level", "cholesterol", "bmi"]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        x = x.fillna(x.median())
        report["numeric_univariate_auc"][col] = float(roc_auc_score(y, x))

    if "d_dimer" in df.columns:
        ctab = pd.crosstab(df["d_dimer"], y)
        deterministic = (
            ctab.shape[0] >= 2
            and ((ctab > 0).sum(axis=1) <= 1).all()
        )
        report["d_dimer_deterministic_with_label"] = bool(deterministic)
        report["recommendation"] = (
            "Exclude d_dimer from the primary predictive metadata set; retain only "
            "for descriptive/sensitivity analysis unless independent prospective "
            "measurement and non-deterministic replication can be demonstrated."
        )

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
