from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = {
    "image_only": "configs/v3_image_only.yaml",
    "concat_linear": "configs/v3_concat_linear.yaml",
    "ccrf_linear": "configs/v3_ccrf_linear.yaml",
    "concat": "configs/v3_concat.yaml",
    "scct": "configs/v3_scct.yaml",
    "scct_gate": "configs/v3_scct_gate.yaml",
    "ccrf": "configs/v3_ccrf.yaml",
    "ccrf_sparse": "configs/v3_ccrf_sparse.yaml",
}

METRICS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auc",
    "pr_auc",
    "mcc",
    "kappa",
]


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", default="generated_index")
    ap.add_argument("--experiment", choices=EXPERIMENTS.keys(), required=True)
    ap.add_argument("--results-root", default="results_v3")
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--end-fold", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    root = Path(args.results_root) / args.experiment

    for fold in range(args.start_fold, args.end_fold + 1):
        csv_path = Path(args.fold_dir) / f"fold_{fold}.csv"
        out_dir = root / f"fold_{fold}"
        ckpt = out_dir / "best.pt"
        metrics_path = out_dir / "metrics.json"

        if args.skip_existing and metrics_path.exists():
            print(f"Skipping fold {fold}: metrics already exist.")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        run([
            sys.executable,
            "train_v3_classifier.py",
            "--csv",
            csv_path,
            "--config",
            cfg,
            "--checkpoint",
            ckpt,
        ])

        run([
            sys.executable,
            "eval_v3_classifier.py",
            "--csv",
            csv_path,
            "--config",
            cfg,
            "--checkpoint",
            ckpt,
            "--out-dir",
            out_dir,
        ])

    rows = []
    for fold in range(5):
        p = root / f"fold_{fold}" / "metrics.json"
        if p.exists():
            obj = json.loads(p.read_text())
            obj["fold"] = fold
            rows.append(obj)

    if len(rows) != 5:
        print(f"Only {len(rows)}/5 folds completed; summary not finalized.")
        return

    summary = []
    for metric in METRICS:
        vals = np.asarray([r[metric] for r in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        summary.append({
            "metric": metric,
            "mean": mean,
            "std": std,
            "mean±std": f"{mean:.4f} ± {std:.4f}",
        })

    frame = pd.DataFrame(summary)
    frame.to_csv(root / "cv_summary.csv", index=False)
    (root / "cv_summary.json").write_text(
        json.dumps({"folds": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
