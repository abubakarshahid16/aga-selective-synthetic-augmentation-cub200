from __future__ import annotations

import argparse
from pathlib import Path

from utils import ROOT_DIR, ensure_output_dirs, get_logger, predict_with_metadata


def main() -> None:
    """Score synthetic candidates with a trained classifier."""

    parser = argparse.ArgumentParser(description="Score synthetic images with a classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates" / "synthetic_manifest.csv")
    parser.add_argument("--synthetic-root", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_classifier_scores.csv")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("score_with_classifier", dirs["logs"] / "score_with_classifier.log")
    scores = predict_with_metadata(
        args.checkpoint,
        args.synthetic_manifest,
        args.image_root,
        args.batch_size,
        args.num_workers,
        args.image_size,
        args.synthetic_root,
    )
    scores["classifier_match"] = scores["label"] == scores["predicted_label"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_csv, index=False)
    logger.info("Saved classifier scores for %d synthetic samples to %s", len(scores), args.output_csv)


if __name__ == "__main__":
    main()
