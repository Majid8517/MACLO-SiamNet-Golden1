from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS = {
    "concat_linear": "configs/v3_concat_linear.yaml",
    "ccrf_linear": "configs/v3_ccrf_linear.yaml",
}

METRICS = [
    "accuracy", "sensitivity", "specificity", "precision",
    "f1", "auc", "pr_auc", "mcc", "kappa"
]


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Run the frozen repeated 5-fold confirmatory evaluation."
    )
    ap.add_argument("--fold-root", default="confirmatory_folds")
    ap.add_argument("--results-root", default="results_confirmatory")
    ap.add_argument(
        "--experiments",
        nargs="+",
        default=["concat_linear", "ccrf_linear"],
        choices=EXPERIMENTS.keys(),
    )
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.fold_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_repeats = int(manifest["n_repeats"])

    for repeat_idx in range(n_repeats):
        repeat_fold_dir = Path(args.fold_root) / f"repeat_{repeat_idx:02d}"

        for experiment in args.experiments:
            cfg = EXPERIMENTS[experiment]
            exp_root = (
                Path(args.results_root)
                / f"repeat_{repeat_idx:02d}"
                / experiment
            )

            print(
                f"\n=== repeat {repeat_idx:02d} | {experiment} ===",
                flush=True,
            )

            for fold in range(5):
                csv_path = repeat_fold_dir / f"fold_{fold}.csv"
                out_dir = exp_root / f"fold_{fold}"
                ckpt = out_dir / "best.pt"
                metrics_path = out_dir / "metrics.json"

                if args.skip_existing and metrics_path.exists():
                    print(f"Skipping repeat {repeat_idx:02d} fold {fold}: exists.")
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                run([
                    sys.executable,
                    "train_v3_classifier.py",
                    "--csv", csv_path,
                    "--config", cfg,
                    "--checkpoint", ckpt,
                ])
                run([
                    sys.executable,
                    "eval_v3_classifier.py",
                    "--csv", csv_path,
                    "--config", cfg,
                    "--checkpoint", ckpt,
                    "--out-dir", out_dir,
                ])

            # Per-repeat summary.
            rows = []
            for fold in range(5):
                p = exp_root / f"fold_{fold}" / "metrics.json"
                if not p.exists():
                    raise FileNotFoundError(p)
                obj = json.loads(p.read_text(encoding="utf-8"))
                obj["fold"] = fold
                rows.append(obj)

            summary_rows = []
            for metric in METRICS:
                vals = np.asarray([r[metric] for r in rows], dtype=float)
                summary_rows.append({
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "mean±std": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}",
                })
            pd.DataFrame(summary_rows).to_csv(
                exp_root / "cv_summary.csv", index=False
            )

    # Global summary across all 25 folds for each experiment.
    for experiment in args.experiments:
        all_rows = []
        for repeat_idx in range(n_repeats):
            for fold in range(5):
                p = (
                    Path(args.results_root)
                    / f"repeat_{repeat_idx:02d}"
                    / experiment
                    / f"fold_{fold}"
                    / "metrics.json"
                )
                obj = json.loads(p.read_text(encoding="utf-8"))
                obj["repeat"] = repeat_idx
                obj["fold"] = fold
                all_rows.append(obj)

        summary = []
        for metric in METRICS:
            vals = np.asarray([r[metric] for r in all_rows], dtype=float)
            summary.append({
                "metric": metric,
                "mean_25fold": float(np.mean(vals)),
                "std_25fold": float(np.std(vals, ddof=1)),
                "mean±std": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}",
            })

        out_dir = Path(args.results_root) / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(
            out_dir / "confirmatory_25fold_summary.csv",
            index=False,
        )
        (out_dir / "confirmatory_25fold_summary.json").write_text(
            json.dumps({"folds": all_rows, "summary": summary}, indent=2),
            encoding="utf-8",
        )
        print(f"\n=== FINAL 25-FOLD SUMMARY | {experiment} ===")
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
