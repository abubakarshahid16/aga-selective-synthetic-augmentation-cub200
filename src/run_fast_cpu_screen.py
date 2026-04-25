from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils import ROOT_DIR, ensure_output_dirs, get_logger, save_json


def run_command(command: List[str], logger) -> None:
    """Run one subprocess and stop on failure."""

    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True)


def load_metrics(path: Path) -> Dict[str, object]:
    """Load a metrics JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Run a fast CPU screening benchmark using frozen pretrained backbones."""

    parser = argparse.ArgumentParser(description="Run a fast CPU screening benchmark.")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--project-src", type=Path, default=ROOT_DIR / "src")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument(
        "--all-synthetic-manifest",
        type=Path,
        default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates" / "synthetic_manifest.csv",
    )
    parser.add_argument(
        "--selected-synthetic-manifest",
        type=Path,
        default=ROOT_DIR / "outputs" / "tables" / "selected_synthetic_manifest.csv",
    )
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("run_fast_cpu_screen", dirs["logs"] / "run_fast_cpu_screen.log")

    experiments = [
        {
            "name": "fastscreen_exp1_real_only",
            "kind": "baseline",
        },
        {
            "name": "fastscreen_exp2_real_plus_all_synthetic",
            "kind": "augmented",
            "synthetic_manifest": args.all_synthetic_manifest,
        },
        {
            "name": "fastscreen_exp3_real_plus_selected_synthetic",
            "kind": "augmented",
            "synthetic_manifest": args.selected_synthetic_manifest,
        },
    ]

    rows = []
    for experiment in experiments:
        metrics_path = dirs["tables"] / f"{experiment['name']}_metrics.json"
        if args.resume and metrics_path.exists():
            logger.info("Resuming: found metrics for %s.", experiment["name"])
            metrics = load_metrics(metrics_path)
        else:
            if experiment["kind"] == "baseline":
                command = [
                    args.python,
                    str(args.project_src / "train_baseline.py"),
                    "--prepared-dir",
                    str(args.prepared_dir),
                    "--image-root",
                    str(args.image_root),
                    "--experiment-name",
                    experiment["name"],
                    "--model-name",
                    args.model_name,
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--image-size",
                    str(args.image_size),
                    "--lr",
                    str(args.lr),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--label-smoothing",
                    str(args.label_smoothing),
                    "--seed",
                    str(args.seed),
                    "--freeze-backbone",
                    "--train-fraction",
                    str(args.train_fraction),
                    "--val-fraction",
                    str(args.val_fraction),
                    "--test-fraction",
                    str(args.test_fraction),
                ]
            else:
                command = [
                    args.python,
                    str(args.project_src / "train_with_augmented_data.py"),
                    "--prepared-dir",
                    str(args.prepared_dir),
                    "--image-root",
                    str(args.image_root),
                    "--synthetic-manifest",
                    str(experiment["synthetic_manifest"]),
                    "--synthetic-root",
                    str(args.all_synthetic_manifest.parent),
                    "--experiment-name",
                    experiment["name"],
                    "--model-name",
                    args.model_name,
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--image-size",
                    str(args.image_size),
                    "--lr",
                    str(args.lr),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--label-smoothing",
                    str(args.label_smoothing),
                    "--seed",
                    str(args.seed),
                    "--freeze-backbone",
                    "--train-fraction",
                    str(args.train_fraction),
                    "--val-fraction",
                    str(args.val_fraction),
                    "--test-fraction",
                    str(args.test_fraction),
                ]
            run_command(command, logger)
            metrics = load_metrics(metrics_path)

        final_summary = metrics["final_summary"] if "final_summary" in metrics else metrics
        rows.append(
            {
                "experiment_name": experiment["name"],
                "test_accuracy": final_summary["test_accuracy"],
                "best_val_accuracy": final_summary["best_val_accuracy"],
                "test_macro_f1": final_summary.get("test_macro_f1"),
                "test_weighted_f1": final_summary.get("test_weighted_f1"),
                "num_epochs_ran": final_summary.get("num_epochs_ran"),
                "freeze_backbone": True,
                "train_fraction": args.train_fraction,
                "image_size": args.image_size,
                "seed": args.seed,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = dirs["tables"] / "fast_cpu_screen_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_json(
        {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "seed": args.seed,
            "freeze_backbone": True,
            "train_fraction": args.train_fraction,
            "summary_csv": str(summary_path),
        },
        dirs["tables"] / "fast_cpu_screen_config.json",
    )
    logger.info("Saved fast CPU screening summary to %s", summary_path)


if __name__ == "__main__":
    main()
