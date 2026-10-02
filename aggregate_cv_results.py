from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


METRICS = ["accuracy","sensitivity","specificity","precision","f1","auc","pr_auc","mcc","kappa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for fold in range(5):
        p = root / args.experiment / f"fold_{fold}" / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        obj["fold"] = fold
        rows.append(obj)

    summary = {}
    for key in METRICS:
        vals = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
        }

    out = Path(args.out) if args.out else root / args.experiment / "cv_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "mean±std"])
        for key in METRICS:
            m, s = summary[key]["mean"], summary[key]["std"]
            w.writerow([key, m, s, f"{m:.4f} ± {s:.4f}"])

    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"folds": rows, "summary": summary}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
