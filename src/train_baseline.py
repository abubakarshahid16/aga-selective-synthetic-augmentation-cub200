from __future__ import annotations

import argparse
from pathlib import Path

from utils import ROOT_DIR, TrainConfig, ensure_output_dirs, fit_classifier, get_logger, plot_training_history
from utils import load_class_names, save_confusion_and_reports


def main() -> None:
    """Train the real-only baseline classifier."""

    parser = argparse.ArgumentParser(description="Train a real-only baseline on CUB-200-2011.")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--experiment-name", type=str, default="exp1_real_only")
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
    config = TrainConfig(
        experiment_name=args.experiment_name,
        train_csv=args.prepared_dir / "train.csv",
        val_csv=args.prepared_dir / "val.csv",
        test_csv=args.prepared_dir / "test.csv",
        image_root=args.image_root,
        synthetic_root=None,
        model_name=args.model_name,
        num_classes=len(class_names),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        image_size=args.image_size,
        use_amp=args.use_amp,
        seed=args.seed,
        checkpoint_path=dirs["checkpoints"] / f"{args.experiment_name}.pt",
        history_csv=dirs["tables"] / f"{args.experiment_name}_history.csv",
        metrics_json=dirs["tables"] / f"{args.experiment_name}_metrics.json",
        class_names_json=args.prepared_dir / "class_names.json",
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=bool(args.freeze_backbone),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        grad_accum_steps=int(args.grad_accum_steps),
        max_grad_norm=float(args.max_grad_norm),
        resume=bool(args.resume),
        latest_checkpoint_path=dirs["checkpoints"] / f"{args.experiment_name}_latest.pt",
        eval_tta=bool(args.eval_tta),
        use_randaugment=bool(args.use_randaugment),
        random_erasing_prob=float(args.random_erasing_prob),
        skip_oom_batches=bool(args.skip_oom_batches),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
    )
    results = fit_classifier(config, logger)
    plot_training_history(config.history_csv, dirs["plots"], args.experiment_name)
    save_confusion_and_reports(
        results["test_metrics"]["labels"],
        results["test_metrics"]["preds"],
        class_names,
        args.experiment_name,
        dirs["confusion_matrices"],
        dirs["tables"],
    )


if __name__ == "__main__":
    main()
