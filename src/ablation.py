from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils import ROOT_DIR, ensure_output_dirs, get_logger


def run_command(command: List[str], logger) -> None:
    """Run a subprocess and raise on failure."""

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


def load_json(path: Path) -> Dict[str, object]:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Run selection ablations and collect comparison tables."""

    parser = argparse.ArgumentParser(description="Run selection ablation experiments.")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--project-src", type=Path, default=ROOT_DIR / "src")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--classifier-scores", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_classifier_scores.csv")
    parser.add_argument("--clip-scores", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_clip_scores.csv")
    parser.add_argument("--dino-pruned", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_dino_pruned.csv")
    parser.add_argument("--model-name", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--clip-threshold", type=float, default=0.20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("ablation", dirs["logs"] / "ablation.log")
    variants: List[Dict[str, object]] = [
        {"name": "ablation_confidence_only", "clip": False, "dino": False, "balance": False, "attribute": False, "ratio": 0.75},
        {"name": "ablation_confidence_clip", "clip": True, "dino": False, "balance": False, "attribute": False, "ratio": 0.75},
        {"name": "ablation_confidence_clip_dino", "clip": True, "dino": True, "balance": False, "attribute": False, "ratio": 0.75},
        {"name": "ablation_confidence_clip_dino_balance", "clip": True, "dino": True, "balance": True, "attribute": False, "ratio": 0.75},
        {"name": "ablation_full_attribute", "clip": True, "dino": True, "balance": True, "attribute": True, "ratio": 0.75},
    ]

    rows = []
    run_rows = []
    for variant in variants:
        output_manifest = dirs["tables"] / f"{variant['name']}_selected.csv"
        if args.resume and output_manifest.exists():
            logger.info("Resuming: found existing selection manifest for %s, skipping rebuild.", variant["name"])
        else:
            build_cmd = [
                args.python,
                str(args.project_src / "build_selected_dataset.py"),
                "--prepared-dir",
                str(args.prepared_dir),
                "--classifier-scores",
                str(args.classifier_scores),
                "--clip-scores",
                str(args.clip_scores),
                "--dino-pruned",
                str(args.dino_pruned),
                "--output-csv",
                str(output_manifest),
                "--max-synth-to-real-ratio",
                str(variant["ratio"]),
                "--confidence-threshold",
                str(args.confidence_threshold),
            ]
            if variant["clip"]:
                build_cmd.extend(["--clip-threshold", str(args.clip_threshold)])
            else:
                build_cmd.append("--disable-clip-filter")
            if not variant["dino"]:
                build_cmd.append("--disable-dino-filter")
            if not variant["balance"]:
                build_cmd.append("--disable-class-balance")
            if variant["attribute"]:
                build_cmd.append("--use-attribute-coverage")
            run_command(build_cmd, logger)

        seed_metrics: List[Dict[str, object]] = []
        for seed in args.seeds:
            run_name = f"{variant['name']}_seed{seed}"
            metrics_path = dirs["tables"] / f"{run_name}_metrics.json"
            if args.resume and metrics_path.exists():
                logger.info("Resuming: found existing metrics for %s, skipping retraining.", run_name)
                metrics = load_json(metrics_path)
            else:
                train_cmd = [
                    args.python,
                    str(args.project_src / "train_with_augmented_data.py"),
                    "--prepared-dir",
                    str(args.prepared_dir),
                    "--image-root",
                    str(args.image_root),
                    "--synthetic-manifest",
                    str(output_manifest),
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
                    "--seed",
                    str(seed),
                ]
                run_command(train_cmd, logger)
                metrics = load_json(metrics_path)
            seed_metrics.append(metrics)
            run_rows.append(
                {
                    "variant_name": variant["name"],
                    "run_name": metrics["experiment_name"],
                    "seed": seed,
                    "warmup_on_real": metrics["warmup_on_real"],
                    "curriculum": metrics["curriculum"],
                    "final_test_accuracy": metrics["final_summary"]["test_accuracy"],
                    "final_best_val_accuracy": metrics["final_summary"]["best_val_accuracy"],
                    "final_test_macro_f1": metrics["final_summary"].get("test_macro_f1"),
                    "final_test_weighted_f1": metrics["final_summary"].get("test_weighted_f1"),
                    "final_checkpoint": metrics["final_checkpoint"],
                }
            )
        test_summary = summarize_metric([float(item["final_summary"]["test_accuracy"]) for item in seed_metrics])
        val_summary = summarize_metric([float(item["final_summary"]["best_val_accuracy"]) for item in seed_metrics])
        macro_f1_summary = summarize_metric([float(item["final_summary"].get("test_macro_f1", 0.0)) for item in seed_metrics])
        rows.append(
            {
                "experiment_name": variant["name"],
                "warmup_on_real": False,
                "curriculum": False,
                "num_seeds": test_summary["n"],
                "final_test_accuracy_mean": test_summary["mean"],
                "final_test_accuracy_std": test_summary["std"],
                "final_test_accuracy_ci95_half_width": test_summary["ci95_half_width"],
                "final_best_val_accuracy_mean": val_summary["mean"],
                "final_best_val_accuracy_std": val_summary["std"],
                "final_test_macro_f1_mean": macro_f1_summary["mean"],
                "final_test_macro_f1_std": macro_f1_summary["std"],
            }
        )

    pd.DataFrame(run_rows).to_csv(dirs["tables"] / "ablation_runs.csv", index=False)
    pd.DataFrame(rows).to_csv(dirs["tables"] / "ablation_summary.csv", index=False)
    logger.info("Ablation summary saved.")


if __name__ == "__main__":
    main()
