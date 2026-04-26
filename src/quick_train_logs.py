from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    ROOT_DIR,
    TrainConfig,
    ensure_output_dirs,
    fit_classifier,
    get_logger,
    load_class_names,
    plot_training_history,
    save_confusion_and_reports,
)


def sample_per_class(frame: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Sample up to a fixed number of rows per class."""

    rows = []
    for _, group in frame.groupby("label", sort=True):
        n = min(len(group), max_per_class)
        rows.append(group.sample(n=n, random_state=seed))
    sampled = pd.concat(rows, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick constrained-compute training pass and save real logs/curves.")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "quick_results")
    parser.add_argument("--experiment-name", type=str, default="quick_resnet18_subset_run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images-per-class", type=int, default=5)
    parser.add_argument("--max-val-images-per-class", type=int, default=2)
    parser.add_argument("--max-test-images-per-class", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output_root)
    logger = get_logger(args.experiment_name, dirs["logs"] / f"{args.experiment_name}.log")
    class_names = load_class_names(args.prepared_dir / "class_names.json")

    train_frame = pd.read_csv(args.prepared_dir / "train.csv")
    val_frame = pd.read_csv(args.prepared_dir / "val.csv")
    test_frame = pd.read_csv(args.prepared_dir / "test.csv")

    quick_train = sample_per_class(train_frame, args.max_train_images_per_class, args.seed)
    quick_val = sample_per_class(val_frame, args.max_val_images_per_class, args.seed)
    quick_test = sample_per_class(test_frame, args.max_test_images_per_class, args.seed)

    quick_train_csv = args.output_root / "training_log.csv.d" / "train_subset.csv"
    quick_val_csv = args.output_root / "training_log.csv.d" / "val_subset.csv"
    quick_test_csv = args.output_root / "training_log.csv.d" / "test_subset.csv"
    quick_train_csv.parent.mkdir(parents=True, exist_ok=True)
    quick_train.to_csv(quick_train_csv, index=False)
    quick_val.to_csv(quick_val_csv, index=False)
    quick_test.to_csv(quick_test_csv, index=False)

    logger.info("Quick constrained run subset sizes | train=%d val=%d test=%d", len(quick_train), len(quick_val), len(quick_test))
    logger.info("Per-class caps | train=%d val=%d test=%d", args.max_train_images_per_class, args.max_val_images_per_class, args.max_test_images_per_class)

    config = TrainConfig(
        experiment_name=args.experiment_name,
        train_csv=quick_train_csv,
        val_csv=quick_val_csv,
        test_csv=quick_test_csv,
        image_root=args.image_root,
        synthetic_root=None,
        model_name="resnet18",
        num_classes=len(class_names),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.epochs,
        label_smoothing=args.label_smoothing,
        image_size=args.image_size,
        use_amp=False,
        seed=args.seed,
        checkpoint_path=dirs["checkpoints"] / "quick_resnet18_checkpoint.pth",
        history_csv=dirs["tables"] / "training_log.csv",
        metrics_json=dirs["tables"] / f"{args.experiment_name}_metrics.json",
        class_names_json=args.prepared_dir / "class_names.json",
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=bool(args.freeze_backbone),
        train_fraction=1.0,
        val_fraction=1.0,
        test_fraction=1.0,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        resume=False,
        latest_checkpoint_path=dirs["checkpoints"] / "quick_resnet18_checkpoint_latest.pth",
        eval_tta=False,
        use_randaugment=False,
        random_erasing_prob=0.0,
        skip_oom_batches=False,
        prefetch_factor=2,
        persistent_workers=False,
    )

    try:
        results = fit_classifier(config, logger)
    except Exception as exc:
        if args.no_pretrained:
            raise
        logger.warning("Pretrained quick run failed (%s). Retrying without pretrained weights.", exc)
        config.pretrained_backbone = False
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

    history = pd.read_csv(config.history_csv)
    history.to_csv(args.output_root / "training_log.csv", index=False)

    loss_src = dirs["plots"] / f"{args.experiment_name}_loss_curves.png"
    acc_src = dirs["plots"] / f"{args.experiment_name}_accuracy_curves.png"
    loss_dst = args.output_root / "fig_loss_curves.png"
    acc_dst = args.output_root / "fig_accuracy_curves.png"
    loss_dst.write_bytes(loss_src.read_bytes())
    acc_dst.write_bytes(acc_src.read_bytes())

    logger.info("Saved quick run artifacts: %s, %s, %s", args.output_root / "training_log.csv", loss_dst, acc_dst)


if __name__ == "__main__":
    main()
