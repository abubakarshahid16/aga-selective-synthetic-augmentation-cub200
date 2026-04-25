from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import ROOT_DIR, ensure_output_dirs, get_logger, load_attributes_map, save_distribution_plot, save_image_grid
from utils import save_partitioned_image_grid


def parse_embedding(text: str) -> np.ndarray:
    """Parse a serialized embedding vector."""

    return np.array(json.loads(text), dtype=np.float32)


def compute_combined_score(row: pd.Series, weights: Dict[str, float]) -> float:
    """Compute a weighted selection score."""

    return (
        weights["confidence"] * float(row["confidence"])
        + weights["clip"] * float(row.get("clip_intended_score", 0.0))
        + weights["margin"] * float(row.get("clip_margin", 0.0))
    )


def enforce_class_balance(frame: pd.DataFrame, base_train_frame: pd.DataFrame, max_ratio: float) -> pd.DataFrame:
    """Limit synthetic additions per class relative to available real training samples."""

    if frame.empty:
        return frame.copy()
    counts = base_train_frame["label"].value_counts().to_dict()
    kept = []
    for label, group in frame.groupby("label"):
        quota = max(1, int(counts.get(label, 1) * max_ratio))
        kept.append(group.nlargest(quota, "combined_score"))
    if not kept:
        return frame.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


def apply_attribute_coverage(
    candidate_frame: pd.DataFrame,
    real_train_frame: pd.DataFrame,
    attributes_csv: Path,
    per_class_quota: int,
) -> pd.DataFrame:
    """Greedily favor samples linked to under-covered attribute profiles via nearest real parents."""

    attribute_map = load_attributes_map(attributes_csv)
    class_selected = []
    for label, group in candidate_frame.groupby("label"):
        real_group = real_train_frame.loc[real_train_frame["label"] == label].copy()
        real_group["attribute_count"] = real_group["image_id"].map(lambda item: len(attribute_map.get(int(item), [])))
        real_group = real_group.sort_values("attribute_count", ascending=False)
        chosen = []
        used_parents = set()
        for _, row in group.sort_values("combined_score", ascending=False).iterrows():
            parent_id = int(row["parent_image_id"])
            parent_attributes = set(attribute_map.get(parent_id, []))
            if parent_id not in used_parents or len(parent_attributes) >= np.median(real_group["attribute_count"]):
                chosen.append(row)
                used_parents.add(parent_id)
            if len(chosen) >= per_class_quota:
                break
        if chosen:
            class_selected.append(pd.DataFrame(chosen))
    return pd.concat(class_selected, ignore_index=True) if class_selected else candidate_frame.iloc[0:0].copy()


def main() -> None:
    """Build the final selected synthetic manifest using the proposed module."""

    parser = argparse.ArgumentParser(description="Build a selected synthetic dataset manifest.")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--classifier-scores", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_classifier_scores.csv")
    parser.add_argument("--clip-scores", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_clip_scores.csv")
    parser.add_argument("--dino-pruned", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_dino_pruned.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--clip-threshold", type=float, default=0.20)
    parser.add_argument("--max-synth-to-real-ratio", type=float, default=0.75)
    parser.add_argument("--use-attribute-coverage", action="store_true")
    parser.add_argument("--disable-clip-filter", action="store_true")
    parser.add_argument("--disable-dino-filter", action="store_true")
    parser.add_argument("--disable-class-balance", action="store_true")
    parser.add_argument("--weights-confidence", type=float, default=1.0)
    parser.add_argument("--weights-clip", type=float, default=0.5)
    parser.add_argument("--weights-margin", type=float, default=0.25)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "outputs" / "tables" / "selected_synthetic_manifest.csv")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("build_selected_dataset", dirs["logs"] / "build_selected_dataset.log")
    real_train = pd.read_csv(args.prepared_dir / "train.csv")
    classifier_frame = pd.read_csv(args.classifier_scores)
    merged = classifier_frame.copy()
    if not args.disable_clip_filter:
        clip_frame = pd.read_csv(args.clip_scores)
        merged = merged.merge(
            clip_frame[["image_id", "clip_intended_score", "clip_best_label", "clip_margin"]],
            on="image_id",
            how="inner",
        )
    else:
        merged["clip_intended_score"] = 0.0
        merged["clip_best_label"] = merged["label"]
        merged["clip_margin"] = 0.0
    if not args.disable_dino_filter:
        dino_frame = pd.read_csv(args.dino_pruned)
        merged = merged.loc[merged["image_id"].isin(dino_frame["image_id"])].copy()
    merged["classifier_match"] = merged["label"] == merged["predicted_label"]
    merged["clip_match"] = merged["label"] == merged["clip_best_label"]
    selection_mask = merged["classifier_match"] & (merged["confidence"] >= args.confidence_threshold)
    if not args.disable_clip_filter:
        selection_mask = selection_mask & merged["clip_match"] & (merged["clip_intended_score"] >= args.clip_threshold)
    merged = merged.loc[selection_mask].copy()

    weights = {
        "confidence": args.weights_confidence,
        "clip": args.weights_clip,
        "margin": args.weights_margin,
    }
    merged["combined_score"] = merged.apply(lambda row: compute_combined_score(row, weights), axis=1)
    if args.disable_class_balance:
        balanced = merged.copy()
    else:
        balanced = enforce_class_balance(merged, real_train, args.max_synth_to_real_ratio)
    if args.use_attribute_coverage:
        attributes_csv = args.prepared_dir / "attributes_image_level.csv"
        if attributes_csv.exists():
            per_class_quota = int(
                max(1, np.ceil(len(balanced) / max(1, balanced["label"].nunique())))
            )
            balanced = apply_attribute_coverage(balanced, real_train, attributes_csv, per_class_quota)
        else:
            logger.warning("Attribute coverage requested but attributes file was not found.")
    balanced = balanced.sort_values(["label", "combined_score"], ascending=[True, False]).reset_index(drop=True)
    balanced.to_csv(args.output_csv, index=False)

    rejected = classifier_frame.loc[~classifier_frame["image_id"].isin(balanced["image_id"])].copy()
    accepted_stats = balanced.groupby("label").size().reset_index(name="accepted_count")
    rejected_stats = rejected.groupby("label").size().reset_index(name="rejected_count")
    accepted_stats.to_csv(dirs["tables"] / "accepted_counts_per_class.csv", index=False)
    rejected_stats.to_csv(dirs["tables"] / "rejected_counts_per_class.csv", index=False)
    pd.DataFrame(
        [
            {
                "num_candidates": int(len(classifier_frame)),
                "num_after_confidence_clip_dino": int(len(merged)),
                "num_selected": int(len(balanced)),
                "num_rejected": int(len(rejected)),
                "selection_rate": float(len(balanced) / max(1, len(classifier_frame))),
                "confidence_threshold": args.confidence_threshold,
                "clip_threshold": None if args.disable_clip_filter else args.clip_threshold,
                "max_synth_to_real_ratio": None if args.disable_class_balance else args.max_synth_to_real_ratio,
                "used_clip_filter": not args.disable_clip_filter,
                "used_dino_filter": not args.disable_dino_filter,
                "used_class_balance": not args.disable_class_balance,
                "use_attribute_coverage": bool(args.use_attribute_coverage),
            }
        ]
    ).to_csv(dirs["tables"] / "accepted_rejected_sample_statistics.csv", index=False)
    if "combined_score" in balanced.columns:
        pd.DataFrame(
            [
                {
                    "combined_score_mean": float(balanced["combined_score"].mean()),
                    "combined_score_std": float(balanced["combined_score"].std(ddof=0)),
                    "confidence_mean": float(balanced["confidence"].mean()),
                    "clip_score_mean": float(balanced["clip_intended_score"].mean()),
                }
            ]
        ).to_csv(dirs["tables"] / "sample_diversity_statistics.csv", index=False)
    save_distribution_plot(real_train, "label", dirs["plots"] / "real_train_distribution.png", "Real Training Distribution")
    save_distribution_plot(balanced, "label", dirs["plots"] / "selected_synthetic_distribution.png", "Selected Synthetic Distribution")
    save_image_grid(
        [Path(path) for path in balanced["image_path"].head(16).tolist()],
        dirs["samples"] / "accepted_synthetic_grid.png",
        "Accepted Synthetic Samples",
    )
    save_image_grid(
        [Path(path) for path in rejected["image_path"].head(16).tolist()],
        dirs["samples"] / "rejected_synthetic_grid.png",
        "Rejected Synthetic Samples",
    )
    save_partitioned_image_grid(
        {
            "real": [Path(path) for path in real_train["image_path"].head(4).tolist()],
            "synthetic": [Path(path) for path in classifier_frame["image_path"].head(4).tolist()],
            "accepted": [Path(path) for path in balanced["image_path"].head(4).tolist()],
            "rejected": [Path(path) for path in rejected["image_path"].head(4).tolist()],
        },
        dirs["samples"] / "real_vs_synthetic_vs_accepted_vs_rejected_grid.png",
        "Real vs Synthetic vs Accepted vs Rejected Samples",
    )
    logger.info("Selected %d synthetic samples from %d candidates.", len(balanced), len(classifier_frame))


if __name__ == "__main__":
    main()
