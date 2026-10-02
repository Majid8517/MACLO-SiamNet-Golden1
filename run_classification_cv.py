from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", default="generated_index")
    ap.add_argument("--experiment", choices=["image_only","image_metadata","metadata_only"], required=True)
    ap.add_argument("--results-root", default="results_v2")
    args = ap.parse_args()

    if args.experiment == "image_only":
        cfg = "configs/classification_image_only.yaml"
    elif args.experiment == "image_metadata":
        cfg = "configs/classification_paper2.yaml"
    else:
        cfg = None

    for fold in range(5):
        csv_path = Path(args.fold_dir) / f"fold_{fold}.csv"
        out_dir = Path(args.results_root) / args.experiment / f"fold_{fold}"

        if args.experiment == "metadata_only":
            run([
                sys.executable, "train_metadata_baseline.py",
                "--csv", csv_path, "--out-dir", out_dir
            ])
            continue

        ckpt = out_dir / "best.pt"
        run([
            sys.executable, "train_v2.py",
            "--csv", csv_path,
            "--config", cfg,
            "--checkpoint", ckpt,
        ])
        run([
            sys.executable, "eval_classification_v2.py",
            "--csv", csv_path,
            "--config", cfg,
            "--checkpoint", ckpt,
            "--split", "test",
            "--out-dir", out_dir,
        ])

    run([
        sys.executable, "aggregate_cv_results.py",
        "--results-root", args.results_root,
        "--experiment", args.experiment,
    ])


if __name__ == "__main__":
    main()
