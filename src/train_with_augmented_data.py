from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils import ROOT_DIR, TrainConfig, append_histories, compute_per_class_delta, ensure_output_dirs
from utils import fit_classifier, get_logger, load_class_names, load_json, merge_real_and_synthetic, plot_training_history
from utils import save_confusion_and_reports, save_json, summarize_manifest


def train_stage(
    experiment_name: str,
    stage_name: str,
    train_csv: Path,
    prepared_dir: Path,
    image_root: Path,
    synthetic_root: Path | None,
    class_names: List[str],
    args,
    dirs: Dict[str, Path],
    logger,
    initial_checkpoint: Path | None,
    pretrained_backbone: bool,
) -> Dict[str, object]:
    """Train one experiment stage and return saved artifacts."""

    metrics_json = dirs["tables"] / f"{experiment_name}_{stage_name}_metrics.json"
    history_csv = dirs["tables"] / f"{experiment_name}_{stage_name}_history.csv"
    checkpoint_path = dirs["checkpoints"] / f"{experiment_name}_{stage_name}.pt"
    if args.resume and metrics_json.exists() and history_csv.exists() and checkpoint_path.exists():
        logger.info("Skipping completed stage %s because resume artifacts already exist.", f"{experiment_name}_{stage_name}")
        return {
            "history": pd.read_csv(history_csv),
            "summary": load_json(metrics_json),
            "test_metrics": {"labels": [], "preds": []},
            "config": None,
            "skipped_existing": True,
        }

    config = TrainConfig(
        experiment_name=f"{experiment_name}_{stage_name}",
        train_csv=train_csv,
        val_csv=prepared_dir / "val.csv",
        test_csv=prepared_dir / "test.csv",
        image_root=image_root,
        synthetic_root=synthetic_root,
        model_name=args.model_name,
        num_classes=len(class_names),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs if stage_name != "warmup" else args.warmup_epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        image_size=args.image_size,
        use_amp=args.use_amp,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        history_csv=history_csv,
        metrics_json=metrics_json,
        class_names_json=prepared_dir / "class_names.json",
        initial_checkpoint=initial_checkpoint,
        pretrained_backbone=pretrained_backbone,
        stage_name=stage_name,
        freeze_backbone=bool(args.freeze_backbone),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        grad_accum_steps=int(args.grad_accum_steps),
        max_grad_norm=float(args.max_grad_norm),
        resume=bool(args.resume),
        latest_checkpoint_path=dirs["checkpoints"] / f"{experiment_name}_{stage_name}_latest.pt",
        eval_tta=bool(args.eval_tta),
        use_randaugment=bool(args.use_randaugment),
        random_erasing_prob=float(args.random_erasing_prob),
        skip_oom_batches=bool(args.skip_oom_batches),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
    )
    return fit_classifier(config, logger)


def main() -> None:
    """Train on real plus synthetic data with optional warm-up and curriculum scheduling."""

    parser = argparse.ArgumentParser(description="Train CUB with real and synthetic data.")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates")
    parser.add_argument("--experiment-name", type=str, default="exp_augmented")
    parser.add_argument("--model-name", type=str, choices=["resnet18", "efficientnet_b0", "convnext_tiny"], default="convnext_tiny")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--warmup-on-real", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--curriculum-ratio", type=float, default=0.5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-tta", action="store_true")
    parser.add_argument("--use-randaugment", action="store_true")
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--skip-oom-batches", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger(args.experiment_name, dirs["logs"] / f"{args.experiment_name}.log")
    class_names = load_class_names(args.prepared_dir / "class_names.json")

    real_train_csv = args.prepared_dir / "train.csv"
    synthetic_frame = pd.read_csv(args.synthetic_manifest).copy()
    synthetic_frame["source"] = "synthetic"
    phase_histories: List[pd.DataFrame] = []
    phase_summaries: List[Dict[str, object]] = []
    latest_checkpoint: Path | None = None

    if args.warmup_on_real:
        warmup_results = train_stage(
            args.experiment_name,
            "warmup",
            real_train_csv,
            args.prepared_dir,
            args.image_root,
            None,
            class_names,
            args,
            dirs,
            logger,
            initial_checkpoint=None,
            pretrained_backbone=not args.no_pretrained,
        )
        phase_histories.append(warmup_results["history"])
        phase_summaries.append(warmup_results["summary"])
        latest_checkpoint = dirs["checkpoints"] / f"{args.experiment_name}_warmup.pt"

    if args.curriculum and len(synthetic_frame) > 0:
        synthetic_frame = synthetic_frame.sort_values(
            "combined_score" if "combined_score" in synthetic_frame.columns else "confidence",
            ascending=False,
        )
        curriculum_count = max(1, int(len(synthetic_frame) * args.curriculum_ratio))
        phase_one_synth = synthetic_frame.iloc[:curriculum_count].copy()
        phase_two_synth = synthetic_frame.copy()
        phase_one_csv = dirs["tables"] / f"{args.experiment_name}_curriculum_phase1_synthetic.csv"
        phase_two_csv = dirs["tables"] / f"{args.experiment_name}_curriculum_phase2_synthetic.csv"
        phase_one_synth.to_csv(phase_one_csv, index=False)
        phase_two_synth.to_csv(phase_two_csv, index=False)

        phase_one_train_csv = dirs["tables"] / f"{args.experiment_name}_curriculum_phase1_train.csv"
        phase_two_train_csv = dirs["tables"] / f"{args.experiment_name}_curriculum_phase2_train.csv"
        phase_one_train = merge_real_and_synthetic(real_train_csv, phase_one_csv, phase_one_train_csv)
        phase_two_train = merge_real_and_synthetic(real_train_csv, phase_two_csv, phase_two_train_csv)

        phase_one_results = train_stage(
            args.experiment_name,
            "curriculum_phase1",
            phase_one_train_csv,
            args.prepared_dir,
            args.image_root,
            args.synthetic_root,
            class_names,
            args,
            dirs,
            logger,
            initial_checkpoint=latest_checkpoint,
            pretrained_backbone=(latest_checkpoint is None and not args.no_pretrained),
        )
        phase_histories.append(phase_one_results["history"])
        phase_summaries.append(
            {
                **phase_one_results["summary"],
                "manifest_summary": summarize_manifest(phase_one_train, "curriculum_phase1"),
            }
        )
        latest_checkpoint = dirs["checkpoints"] / f"{args.experiment_name}_curriculum_phase1.pt"

        phase_two_results = train_stage(
            args.experiment_name,
            "curriculum_phase2",
            phase_two_train_csv,
            args.prepared_dir,
            args.image_root,
            args.synthetic_root,
            class_names,
            args,
            dirs,
            logger,
            initial_checkpoint=latest_checkpoint,
            pretrained_backbone=False,
        )
        final_results = phase_two_results
        final_train_manifest = phase_two_train
        phase_histories.append(phase_two_results["history"])
        phase_summaries.append(
            {
                **phase_two_results["summary"],
                "manifest_summary": summarize_manifest(phase_two_train, "curriculum_phase2"),
            }
        )
        latest_checkpoint = dirs["checkpoints"] / f"{args.experiment_name}_curriculum_phase2.pt"
    else:
        synthetic_subset_csv = dirs["tables"] / f"{args.experiment_name}_synthetic_subset.csv"
        synthetic_frame.to_csv(synthetic_subset_csv, index=False)
        merged_train_csv = dirs["tables"] / f"{args.experiment_name}_train_manifest.csv"
        final_train_manifest = merge_real_and_synthetic(real_train_csv, synthetic_subset_csv, merged_train_csv)
        final_results = train_stage(
            args.experiment_name,
            "main",
            merged_train_csv,
            args.prepared_dir,
            args.image_root,
            args.synthetic_root if len(synthetic_frame) > 0 else None,
            class_names,
            args,
            dirs,
            logger,
            initial_checkpoint=latest_checkpoint,
            pretrained_backbone=(latest_checkpoint is None and not args.no_pretrained),
        )
        phase_histories.append(final_results["history"])
        phase_summaries.append(
            {
                **final_results["summary"],
                "manifest_summary": summarize_manifest(final_train_manifest, "main"),
            }
        )
        latest_checkpoint = dirs["checkpoints"] / f"{args.experiment_name}_main.pt"

    combined_history_csv = dirs["tables"] / f"{args.experiment_name}_history.csv"
    append_histories(phase_histories, combined_history_csv)
    plot_training_history(combined_history_csv, dirs["plots"], args.experiment_name)

    final_checkpoint = dirs["checkpoints"] / f"{args.experiment_name}.pt"
    final_checkpoint.write_bytes(latest_checkpoint.read_bytes())
    if final_results["test_metrics"]["labels"] and final_results["test_metrics"]["preds"]:
        save_confusion_and_reports(
            final_results["test_metrics"]["labels"],
            final_results["test_metrics"]["preds"],
            class_names,
            args.experiment_name,
            dirs["confusion_matrices"],
            dirs["tables"],
        )

    baseline_per_class = dirs["tables"] / "exp1_real_only_per_class_accuracy.csv"
    current_per_class = dirs["tables"] / f"{args.experiment_name}_per_class_accuracy.csv"
    if baseline_per_class.exists():
        compute_per_class_delta(
            baseline_per_class,
            current_per_class,
            dirs["tables"] / f"{args.experiment_name}_per_class_gains_losses.csv",
            "baseline",
            args.experiment_name,
        )

    summary = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "warmup_on_real": bool(args.warmup_on_real),
        "curriculum": bool(args.curriculum),
        "curriculum_ratio": float(args.curriculum_ratio),
        "final_checkpoint": str(final_checkpoint),
        "num_final_train_samples": int(len(final_train_manifest)),
        "phase_summaries": phase_summaries,
        "final_summary": final_results["summary"],
    }
    save_json(summary, dirs["tables"] / f"{args.experiment_name}_metrics.json")
    phase_metrics_frame = pd.DataFrame(
        [
            {
                **item,
                "manifest_summary": json.dumps(item.get("manifest_summary", {})),
            }
            for item in phase_summaries
        ]
    )
    phase_metrics_frame.to_csv(dirs["tables"] / f"{args.experiment_name}_phase_metrics.csv", index=False)


if __name__ == "__main__":
    main()
