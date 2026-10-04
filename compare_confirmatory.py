from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar


def load_patient_averaged_predictions(results_root: Path, experiment: str, n_repeats: int):
    rows = []
    for repeat_idx in range(n_repeats):
        for fold in range(5):
            p = (
                results_root
                / f"repeat_{repeat_idx:02d}"
                / experiment
                / f"fold_{fold}"
                / "predictions.csv"
            )
            df = pd.read_csv(p)
            df["repeat"] = repeat_idx
            df["fold"] = fold
            rows.append(df)

    all_pred = pd.concat(rows, ignore_index=True)

    # Each patient must have one OOF prediction per repeat.
    counts = all_pred.groupby("patient_id").size()
    if not (counts == n_repeats).all():
        bad = counts[counts != n_repeats]
        raise ValueError(
            f"Expected {n_repeats} OOF predictions per patient; bad counts: {bad.head().to_dict()}"
        )

    grouped = (
        all_pred.groupby("patient_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
        )
    )
    grouped["y_pred"] = (grouped["y_prob"] >= 0.5).astype(int)
    return grouped


def metric_triplet(df):
    y = df["y_true"].to_numpy(dtype=int)
    p = df["y_prob"].to_numpy(dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def paired_bootstrap(base, cand, n_boot=10000, seed=2026):
    rng = np.random.default_rng(seed)
    y = base["y_true"].to_numpy(dtype=int)
    pb = base["y_prob"].to_numpy(dtype=float)
    pc = cand["y_prob"].to_numpy(dtype=float)
    n = len(y)

    diffs = {"auc": [], "f1": [], "mcc": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue

        pbb = pb[idx]
        pcc = pc[idx]
        ybb = (pbb >= 0.5).astype(int)
        ycc = (pcc >= 0.5).astype(int)

        diffs["auc"].append(roc_auc_score(yy, pcc) - roc_auc_score(yy, pbb))
        diffs["f1"].append(
            f1_score(yy, ycc, zero_division=0)
            - f1_score(yy, ybb, zero_division=0)
        )
        diffs["mcc"].append(
            matthews_corrcoef(yy, ycc)
            - matthews_corrcoef(yy, ybb)
        )

    out = {}
    for key, vals in diffs.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean_diff": float(np.mean(arr)),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "bootstrap_p_two_sided": float(
                2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
            ),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results_confirmatory")
    ap.add_argument("--baseline", default="concat_linear")
    ap.add_argument("--candidate", default="ccrf_linear")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.results_root)
    base = load_patient_averaged_predictions(root, args.baseline, args.repeats)
    cand = load_patient_averaged_predictions(root, args.candidate, args.repeats)

    merged = base.merge(
        cand,
        on="patient_id",
        suffixes=("_base", "_cand"),
        validate="one_to_one",
    )
    if not np.array_equal(
        merged["y_true_base"].to_numpy(),
        merged["y_true_cand"].to_numpy(),
    ):
        raise ValueError("Label mismatch after patient alignment.")

    b = pd.DataFrame({
        "patient_id": merged["patient_id"],
        "y_true": merged["y_true_base"],
        "y_prob": merged["y_prob_base"],
    })
    c = pd.DataFrame({
        "patient_id": merged["patient_id"],
        "y_true": merged["y_true_cand"],
        "y_prob": merged["y_prob_cand"],
    })

    mb = metric_triplet(b)
    mc = metric_triplet(c)

    y = b["y_true"].to_numpy(dtype=int)
    pred_b = (b["y_prob"].to_numpy() >= 0.5).astype(int)
    pred_c = (c["y_prob"].to_numpy() >= 0.5).astype(int)
    correct_b = pred_b == y
    correct_c = pred_c == y

    table = [
        [int(np.sum(correct_b & correct_c)), int(np.sum(correct_b & ~correct_c))],
        [int(np.sum(~correct_b & correct_c)), int(np.sum(~correct_b & ~correct_c))],
    ]
    mc_test = mcnemar(table, exact=True)

    out = {
        "n_patients": int(len(b)),
        "repeats_averaged_per_patient": int(args.repeats),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_patient_averaged_oof": mb,
        "candidate_patient_averaged_oof": mc,
        "paired_patient_bootstrap": paired_bootstrap(b, c),
        "mcnemar": {
            "table": table,
            "statistic": float(mc_test.statistic),
            "p_value": float(mc_test.pvalue),
        },
        "note": (
            "Each patient contributes one averaged OOF probability per model, formed from "
            "one test prediction in each repeated 5-fold CV repeat. Bootstrap resampling is "
            "performed at the patient level."
        ),
    }

    out_path = root / "confirmatory_paired_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
