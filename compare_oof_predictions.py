from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar


def load_oof(root: Path, experiment: str):
    frames = []
    for fold in range(5):
        p = root / experiment / f"fold_{fold}" / "predictions.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df["fold"] = fold
        frames.append(df[["patient_id","y_true","y_prob","y_pred","fold"]])
    out = pd.concat(frames, ignore_index=True)
    if out["patient_id"].duplicated().any():
        raise ValueError(f"Duplicate patient IDs in {experiment}")
    return out.sort_values("patient_id").reset_index(drop=True)


def bootstrap_diff(a, b, metric_fn, n_boot=5000, seed=2025):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            va = metric_fn(a[idx])
            vb = metric_fn(b[idx])
            diffs.append(vb - va)
        except Exception:
            continue
    diffs = np.asarray(diffs, dtype=float)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "bootstrap_p_two_sided": float(
            2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results_v3")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results_root)
    a = load_oof(root, args.baseline)
    b = load_oof(root, args.candidate)

    merged = a.merge(
        b,
        on=["patient_id","y_true"],
        suffixes=("_a","_b"),
        validate="one_to_one",
    )
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("OOF cohorts do not align exactly.")

    y = merged["y_true"].to_numpy(dtype=int)
    pa = merged["y_prob_a"].to_numpy(dtype=float)
    pb = merged["y_prob_b"].to_numpy(dtype=float)
    ya = merged["y_pred_a"].to_numpy(dtype=int)
    yb = merged["y_pred_b"].to_numpy(dtype=int)

    base = {
        "auc": float(roc_auc_score(y, pa)),
        "f1": float(f1_score(y, ya)),
        "mcc": float(matthews_corrcoef(y, ya)),
    }
    cand = {
        "auc": float(roc_auc_score(y, pb)),
        "f1": float(f1_score(y, yb)),
        "mcc": float(matthews_corrcoef(y, yb)),
    }

    idx = np.arange(len(y))
    auc_boot = bootstrap_diff(
        idx, idx,
        lambda ii: roc_auc_score(y[ii], pb[ii]) - roc_auc_score(y[ii], pa[ii]),
        n_boot=args.n_bootstrap,
    )
    # bootstrap_diff above compares its second and first metric values; here the lambda
    # already returns the candidate-baseline difference, so compute directly below instead.
    rng = np.random.default_rng(2025)
    auc_d, f1_d, mcc_d = [], [], []
    for _ in range(args.n_bootstrap):
        ii = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ii])) < 2:
            continue
        auc_d.append(roc_auc_score(y[ii], pb[ii]) - roc_auc_score(y[ii], pa[ii]))
        f1_d.append(f1_score(y[ii], yb[ii]) - f1_score(y[ii], ya[ii]))
        mcc_d.append(matthews_corrcoef(y[ii], yb[ii]) - matthews_corrcoef(y[ii], ya[ii]))

    def summarize(d):
        d = np.asarray(d, dtype=float)
        return {
            "mean_diff": float(np.mean(d)),
            "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "bootstrap_p_two_sided": float(
                min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0)))
            ),
        }

    # McNemar on paired correctness at the prespecified 0.5 threshold.
    correct_a = ya == y
    correct_b = yb == y
    table = np.array([
        [np.sum(correct_a & correct_b), np.sum(correct_a & ~correct_b)],
        [np.sum(~correct_a & correct_b), np.sum(~correct_a & ~correct_b)],
    ])
    mc = mcnemar(table, exact=True)

    report = {
        "n_patients": int(len(y)),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_oof": base,
        "candidate_oof": cand,
        "paired_bootstrap": {
            "auc": summarize(auc_d),
            "f1": summarize(f1_d),
            "mcc": summarize(mcc_d),
        },
        "mcnemar": {
            "table": table.tolist(),
            "statistic": float(mc.statistic),
            "p_value": float(mc.pvalue),
        },
        "note": (
            "OOF predictions from the same patient-level folds are paired by patient_id. "
            "The 0.5 decision threshold is fixed and is not optimized on the test data."
        ),
    }

    out = Path(args.out) if args.out else root / f"compare_{args.baseline}_vs_{args.candidate}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
