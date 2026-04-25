from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils import ROOT_DIR, ensure_output_dirs, get_logger, save_json


def run_command(command: List[str], logger) -> None:
    """Run a subprocess and fail loudly if a publication benchmark step breaks."""

    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True)


def summarize_metric(values: List[float]) -> Dict[str, float]:
    """Summarize repeated-run metrics for publication tables."""

    series = pd.Series(values, dtype=float)
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    stderr = std / math.sqrt(len(series)) if len(series) > 1 else 0.0
    return {
        "mean": float(series.mean()),
        "std": std,
        "min": float(series.min()),
        "max": float(series.max()),
        "n": int(len(series)),
        "ci95_half_width": 1.96 * stderr,
    }


def load_metrics(path: Path) -> Dict[str, object]:
    """Load saved experiment metrics."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_final_summary(metrics: Dict[str, object]) -> Dict[str, object]:
    """Normalize saved metrics into a final-summary view."""

    return metrics["final_summary"] if "final_summary" in metrics else metrics


def main() -> None:
    """Run multi-seed benchmark experiments suitable for publication tables."""

    parser = argparse.ArgumentParser(description="Run publication-grade benchmark experiments across multiple seeds.")
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
    parser.add_argument("--model-name", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--selected-warmup-on-real", action="store_true")
    parser.add_argument("--selected-curriculum", action="store_true")
    parser.add_argument("--selected-curriculum-ratio", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("run_publication_benchmark", dirs["logs"] / "run_publication_benchmark.log")

    experiments = [
        {
            "name": "exp1_real_only",
            "kind": "baseline",
        },
        {
            "name": "exp2_real_plus_all_synthetic",
            "kind": "augmented",
            "synthetic_manifest": args.all_synthetic_manifest,
            "warmup_on_real": False,
            "curriculum": False,
            "curriculum_ratio": 0.5,
        },
        {
            "name": "exp3_real_plus_selected_synthetic",
            "kind": "augmented",
            "synthetic_manifest": args.selected_synthetic_manifest,
            "warmup_on_real": bool(args.selected_warmup_on_real),
            "curriculum": bool(args.selected_curriculum),
            "curriculum_ratio": float(args.selected_curriculum_ratio),
        },
    ]

    run_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for experiment in experiments:
        experiment_metrics: List[Dict[str, object]] = []
        for seed in args.seeds:
            run_name = f"{experiment['name']}_seed{seed}"
            metrics_path = dirs["tables"] / f"{run_name}_metrics.json"
            if args.resume and metrics_path.exists():
                logger.info("Resuming: found existing metrics for %s, skipping retraining.", run_name)
                metrics = load_metrics(metrics_path)
                experiment_metrics.append(metrics)
                final_summary = extract_final_summary(metrics)
                run_rows.append(
                    {
                        "experiment_name": experiment["name"],
                        "run_name": run_name,
                        "seed": seed,
                        "model_name": args.model_name,
                        "test_accuracy": final_summary["test_accuracy"],
                        "best_val_accuracy": final_summary["best_val_accuracy"],
                        "test_macro_f1": final_summary.get("test_macro_f1"),
                        "test_weighted_f1": final_summary.get("test_weighted_f1"),
                        "checkpoint_path": final_summary["checkpoint_path"],
                    }
                )
                continue
            if experiment["kind"] == "baseline":
                command = [
                    args.python,
                    str(args.project_src / "train_baseline.py"),
                    "--prepared-dir",
                    str(args.prepared_dir),
                    "--image-root",
                    str(args.image_root),
                    "--experiment-name",
                    run_name,
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
                    str(seed),
                ]
                if args.use_amp:
                    command.append("--use-amp")
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
                    run_name,
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
                    str(seed),
                    "--curriculum-ratio",
                    str(experiment["curriculum_ratio"]),
                ]
                if args.use_amp:
                    command.append("--use-amp")
                if experiment["warmup_on_real"]:
                    command.append("--warmup-on-real")
                if experiment["curriculum"]:
                    command.append("--curriculum")
            run_command(command, logger)
            metrics = load_metrics(metrics_path)
            experiment_metrics.append(metrics)

            final_summary = extract_final_summary(metrics)
            run_rows.append(
                {
                    "experiment_name": experiment["name"],
                    "run_name": run_name,
                    "seed": seed,
                    "model_name": args.model_name,
                    "test_accuracy": final_summary["test_accuracy"],
                    "best_val_accuracy": final_summary["best_val_accuracy"],
                    "test_macro_f1": final_summary.get("test_macro_f1"),
                    "test_weighted_f1": final_summary.get("test_weighted_f1"),
                    "checkpoint_path": final_summary["checkpoint_path"],
                }
            )

        accuracy_summary = summarize_metric(
            [float(extract_final_summary(item)["test_accuracy"]) for item in experiment_metrics]
        )
        val_summary = summarize_metric(
            [float(extract_final_summary(item)["best_val_accuracy"]) for item in experiment_metrics]
        )
        macro_f1_summary = summarize_metric(
            [float(extract_final_summary(item).get("test_macro_f1", 0.0)) for item in experiment_metrics]
        )
        weighted_f1_summary = summarize_metric(
            [float(extract_final_summary(item).get("test_weighted_f1", 0.0)) for item in experiment_metrics]
        )

        summary_rows.append(
            {
                "experiment_name": experiment["name"],
                "model_name": args.model_name,
                "num_seeds": accuracy_summary["n"],
                "test_accuracy_mean": accuracy_summary["mean"],
                "test_accuracy_std": accuracy_summary["std"],
                "test_accuracy_ci95_half_width": accuracy_summary["ci95_half_width"],
                "best_val_accuracy_mean": val_summary["mean"],
                "best_val_accuracy_std": val_summary["std"],
                "test_macro_f1_mean": macro_f1_summary["mean"],
                "test_macro_f1_std": macro_f1_summary["std"],
                "test_weighted_f1_mean": weighted_f1_summary["mean"],
                "test_weighted_f1_std": weighted_f1_summary["std"],
            }
        )

    runs_path = dirs["tables"] / "publication_benchmark_runs.csv"
    summary_path = dirs["tables"] / "publication_benchmark_summary.csv"
    pd.DataFrame(run_rows).to_csv(runs_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    save_json(
        {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "use_amp": bool(args.use_amp),
            "seeds": args.seeds,
            "selected_warmup_on_real": bool(args.selected_warmup_on_real),
            "selected_curriculum": bool(args.selected_curriculum),
            "selected_curriculum_ratio": float(args.selected_curriculum_ratio),
            "runs_csv": str(runs_path),
            "summary_csv": str(summary_path),
        },
        dirs["tables"] / "publication_benchmark_config.json",
    )
    logger.info("Saved publication benchmark outputs to %s and %s", runs_path, summary_path)


if __name__ == "__main__":
    main()
