from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

from utils import ROOT_DIR, ensure_output_dirs, get_logger, save_json


def run_command(command: List[str], logger) -> None:
    """Run one pipeline command."""

    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True)


def file_ready(path: Path) -> bool:
    """Return whether a file exists and is non-empty."""

    return path.exists() and path.stat().st_size > 0


def main() -> None:
    """Run a Kaggle-friendly end-to-end pipeline with restart safety."""

    parser = argparse.ArgumentParser(description="Run a Kaggle-safe, resume-friendly optimized experiment pipeline.")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--synthetic-root", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates")
    parser.add_argument("--model-name", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.03)
    parser.add_argument("--clip-threshold", type=float, default=0.22)
    parser.add_argument("--max-synth-to-real-ratio", type=float, default=0.75)
    parser.add_argument("--curriculum-ratio", type=float, default=0.5)
    parser.add_argument("--skip-exp2", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("run_kaggle_best_pipeline", dirs["logs"] / "run_kaggle_best_pipeline.log")

    exp1_name = "kaggle_exp1_real_only"
    exp2_name = "kaggle_exp2_real_plus_all_synthetic"
    exp3_name = "kaggle_exp3_real_plus_selected_synthetic"

    baseline_checkpoint = dirs["checkpoints"] / f"{exp1_name}.pt"
    classifier_scores = dirs["tables"] / "synthetic_classifier_scores.csv"
    clip_scores = dirs["tables"] / "synthetic_clip_scores.csv"
    dino_pruned = dirs["tables"] / "synthetic_dino_pruned.csv"
    selected_manifest = dirs["tables"] / "selected_synthetic_manifest.csv"
    final_eval_json = dirs["tables"] / f"{exp3_name}_kaggle_eval_evaluation_metrics.json"

    commands: List[Dict[str, object]] = []

    baseline_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "train_baseline.py"),
        "--prepared-dir",
        str(args.prepared_dir),
        "--image-root",
        str(args.image_root),
        "--experiment-name",
        exp1_name,
        "--model-name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
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
        "--use-amp",
        "--resume",
        "--eval-tta",
        "--use-randaugment",
        "--random-erasing-prob",
        "0.10",
        "--skip-oom-batches",
        "--prefetch-factor",
        "2",
        "--persistent-workers",
    ]
    commands.append({"step": "baseline_train", "command": baseline_cmd})
    if not file_ready(baseline_checkpoint):
        run_command(baseline_cmd, logger)

    classifier_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "score_with_classifier.py"),
        "--checkpoint",
        str(baseline_checkpoint),
        "--synthetic-manifest",
        str(args.synthetic_root / "synthetic_manifest.csv"),
        "--synthetic-root",
        str(args.synthetic_root),
        "--output-csv",
        str(classifier_scores),
        "--num-workers",
        str(args.num_workers),
    ]
    commands.append({"step": "classifier_scoring", "command": classifier_cmd})
    if not file_ready(classifier_scores):
        run_command(classifier_cmd, logger)

    clip_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "score_with_clip.py"),
        "--manifest",
        str(args.synthetic_root / "synthetic_manifest.csv"),
        "--class-names",
        str(args.prepared_dir / "class_names.json"),
        "--output-csv",
        str(clip_scores),
        "--batch-size",
        "32",
    ]
    commands.append({"step": "clip_scoring", "command": clip_cmd})
    if not file_ready(clip_scores):
        run_command(clip_cmd, logger)

    dino_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "prune_with_dino.py"),
        "--manifest",
        str(args.synthetic_root / "synthetic_manifest.csv"),
        "--batch-size",
        "32",
        "--num-workers",
        str(args.num_workers),
        "--similarity-threshold",
        "0.94",
        "--output-csv",
        str(dino_pruned),
        "--embeddings-csv",
        str(dirs["tables"] / "synthetic_dino_embeddings.csv"),
    ]
    commands.append({"step": "dino_pruning", "command": dino_cmd})
    if not file_ready(dino_pruned):
        run_command(dino_cmd, logger)

    select_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "build_selected_dataset.py"),
        "--prepared-dir",
        str(args.prepared_dir),
        "--classifier-scores",
        str(classifier_scores),
        "--clip-scores",
        str(clip_scores),
        "--dino-pruned",
        str(dino_pruned),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--clip-threshold",
        str(args.clip_threshold),
        "--max-synth-to-real-ratio",
        str(args.max_synth_to_real_ratio),
        "--use-attribute-coverage",
        "--output-csv",
        str(selected_manifest),
    ]
    commands.append({"step": "selected_manifest", "command": select_cmd})
    if not file_ready(selected_manifest):
        run_command(select_cmd, logger)

    if not args.skip_exp2:
        exp2_cmd = [
            args.python,
            str(ROOT_DIR / "src" / "train_with_augmented_data.py"),
            "--prepared-dir",
            str(args.prepared_dir),
            "--image-root",
            str(args.image_root),
            "--synthetic-manifest",
            str(args.synthetic_root / "synthetic_manifest.csv"),
            "--synthetic-root",
            str(args.synthetic_root),
            "--experiment-name",
            exp2_name,
            "--model-name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
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
            "--use-amp",
            "--resume",
            "--eval-tta",
            "--use-randaugment",
            "--random-erasing-prob",
            "0.10",
            "--skip-oom-batches",
            "--prefetch-factor",
            "2",
            "--persistent-workers",
        ]
        commands.append({"step": "exp2_train", "command": exp2_cmd})
        run_command(exp2_cmd, logger)

    exp3_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "train_with_augmented_data.py"),
        "--prepared-dir",
        str(args.prepared_dir),
        "--image-root",
        str(args.image_root),
        "--synthetic-manifest",
        str(selected_manifest),
        "--synthetic-root",
        str(args.synthetic_root),
        "--experiment-name",
        exp3_name,
        "--model-name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
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
        "--use-amp",
        "--resume",
        "--eval-tta",
        "--use-randaugment",
        "--random-erasing-prob",
        "0.10",
        "--skip-oom-batches",
        "--prefetch-factor",
        "2",
        "--persistent-workers",
        "--warmup-on-real",
        "--warmup-epochs",
        "4",
        "--curriculum",
        "--curriculum-ratio",
        str(args.curriculum_ratio),
    ]
    commands.append({"step": "exp3_train", "command": exp3_cmd})
    run_command(exp3_cmd, logger)

    eval_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "evaluate.py"),
        "--checkpoint",
        str(dirs["checkpoints"] / f"{exp3_name}.pt"),
        "--csv-path",
        str(args.prepared_dir / "test.csv"),
        "--image-root",
        str(args.image_root),
        "--class-names",
        str(args.prepared_dir / "class_names.json"),
        "--experiment-name",
        f"{exp3_name}_kaggle_eval",
        "--use-amp",
        "--eval-tta",
    ]
    commands.append({"step": "final_eval", "command": eval_cmd})
    run_command(eval_cmd, logger)

    metric_jsons = [
        str(dirs["tables"] / f"{exp1_name}_metrics.json"),
        str(dirs["tables"] / f"{exp3_name}_metrics.json"),
    ]
    exp2_metrics_path = dirs["tables"] / f"{exp2_name}_metrics.json"
    if file_ready(exp2_metrics_path):
        metric_jsons.insert(1, str(exp2_metrics_path))
    plot_cmd = [
        args.python,
        str(ROOT_DIR / "src" / "plot_results.py"),
        "--metrics-jsons",
        *metric_jsons,
        "--ablation-summary",
        str(dirs["tables"] / "ablation_summary.csv"),
    ]
    commands.append({"step": "plot_results", "command": plot_cmd})
    run_command(plot_cmd, logger)

    proof_payload = {
        "pipeline_name": "kaggle_best_pipeline",
        "settings": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "num_workers": args.num_workers,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "confidence_threshold": args.confidence_threshold,
            "clip_threshold": args.clip_threshold,
            "max_synth_to_real_ratio": args.max_synth_to_real_ratio,
            "curriculum_ratio": args.curriculum_ratio,
        },
        "commands": commands,
        "key_outputs": {
            "baseline_metrics": str(dirs["tables"] / f"{exp1_name}_metrics.json"),
            "exp2_metrics": str(dirs["tables"] / f"{exp2_name}_metrics.json"),
            "exp3_metrics": str(dirs["tables"] / f"{exp3_name}_metrics.json"),
            "selected_manifest": str(selected_manifest),
            "accepted_rejected_stats": str(dirs["tables"] / "accepted_rejected_sample_statistics.csv"),
            "dino_stats": str(dirs["tables"] / "dino_pruning_stats.csv"),
            "final_eval": str(final_eval_json),
            "comparison_plot": str(dirs["plots"] / "experiment_comparison_accuracy.png"),
        },
    }
    save_json(proof_payload, dirs["tables"] / "kaggle_best_pipeline_proof_bundle.json")
    logger.info("Saved proof bundle to %s", dirs["tables"] / "kaggle_best_pipeline_proof_bundle.json")


if __name__ == "__main__":
    main()
