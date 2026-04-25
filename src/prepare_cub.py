from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils import ROOT_DIR, ensure_output_dirs, get_logger, save_json, set_seed


def read_mapping_file(path: Path, columns: List[str]) -> pd.DataFrame:
    """Read a whitespace-separated CUB metadata file."""

    return pd.read_csv(path, sep=r"\s+", names=columns, engine="python")


def build_metadata(raw_dir: Path) -> pd.DataFrame:
    """Assemble the full CUB metadata table from official split files."""

    images = read_mapping_file(raw_dir / "images.txt", ["image_id", "relative_path"])
    labels = read_mapping_file(raw_dir / "image_class_labels.txt", ["image_id", "class_id"])
    splits = read_mapping_file(raw_dir / "train_test_split.txt", ["image_id", "is_train"])
    classes = read_mapping_file(raw_dir / "classes.txt", ["class_id", "class_name"])
    bboxes = read_mapping_file(
        raw_dir / "bounding_boxes.txt",
        ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
    )
    frame = images.merge(labels, on="image_id").merge(splits, on="image_id").merge(bboxes, on="image_id")
    frame = frame.merge(classes, on="class_id")
    frame["label"] = frame["class_id"] - 1
    frame["class_name_readable"] = frame["class_name"].str.split(".", n=1).str[-1].str.replace("_", " ")
    frame["source"] = "real"
    frame["image_path"] = frame["relative_path"].map(lambda rel: str((raw_dir / "images" / rel).resolve()))
    return frame


def parse_attributes(raw_dir: Path, output_dir: Path, logger) -> Path | None:
    """Parse optional CUB image attribute labels into a compact CSV."""

    attr_labels_path = raw_dir / "attributes" / "image_attribute_labels.txt"
    attr_names_candidates = [
        raw_dir / "attributes" / "attributes.txt",
        raw_dir / "attributes.txt",
    ]
    attr_names_path = next((path for path in attr_names_candidates if path.exists()), None)
    if not attr_labels_path.exists() or attr_names_path is None:
        logger.info("Attribute files not found. Attribute-aware coverage will remain optional.")
        return None

    attr_names = pd.read_csv(attr_names_path, sep=r"\s+", names=["attribute_id", "attribute_name"], engine="python")
    attr_map = dict(zip(attr_names["attribute_id"].tolist(), attr_names["attribute_name"].tolist()))
    parsed_rows = []
    with attr_labels_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 3:
                logger.warning("Skipping malformed attribute row %d with fewer than 3 fields.", line_number)
                continue
            parsed_rows.append(
                {
                    "image_id": int(parts[0]),
                    "attribute_id": int(parts[1]),
                    "is_present": int(parts[2]),
                }
            )
    attr_labels = pd.DataFrame(parsed_rows)

    rows = []
    for image_id, group in attr_labels.groupby("image_id"):
        active = group.loc[group["is_present"] == 1, "attribute_id"].astype(int).tolist()
        rows.append(
            {
                "image_id": int(image_id),
                "active_attributes": json.dumps(active),
                "active_attribute_names": json.dumps([attr_map[item] for item in active]),
            }
        )
    output_path = output_dir / "attributes_image_level.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    save_json({str(k): v for k, v in attr_map.items()}, output_dir / "attribute_names.json")
    return output_path


def split_train_val(train_frame: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the official training set into train and validation subsets."""

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    indices = np.arange(len(train_frame))
    labels = train_frame["label"].to_numpy()
    train_idx, val_idx = next(splitter.split(indices, labels))
    return train_frame.iloc[train_idx].copy(), train_frame.iloc[val_idx].copy()


def main() -> None:
    """Prepare CUB metadata and optional validation splits."""

    parser = argparse.ArgumentParser(description="Prepare CUB-200-2011 metadata and splits.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs()
    set_seed(args.seed)
    logger = get_logger("prepare_cub", ROOT_DIR / "outputs" / "logs" / "prepare_cub.log")

    if not args.raw_dir.exists():
        raise FileNotFoundError(
            f"CUB raw directory not found at {args.raw_dir}. Download the official dataset and place it there."
        )

    frame = build_metadata(args.raw_dir)
    train_frame = frame.loc[frame["is_train"] == 1].copy()
    test_frame = frame.loc[frame["is_train"] == 0].copy()
    split_train, split_val = split_train_val(train_frame, args.val_ratio, args.seed)

    for subset_name, subset_frame in {
        "all_metadata": frame,
        "train": split_train,
        "val": split_val,
        "test": test_frame,
    }.items():
        subset_path = args.output_dir / f"{subset_name}.csv"
        subset_frame.to_csv(subset_path, index=False)
        logger.info("Saved %s with %d rows to %s", subset_name, len(subset_frame), subset_path)

    class_names = {
        str(int(row["label"])): row["class_name_readable"]
        for _, row in frame[["label", "class_name_readable"]].drop_duplicates().sort_values("label").iterrows()
    }
    save_json(class_names, args.output_dir / "class_names.json")
    attributes_path = parse_attributes(args.raw_dir, args.output_dir, logger)
    summary = {
        "num_total": int(len(frame)),
        "num_train": int(len(split_train)),
        "num_val": int(len(split_val)),
        "num_test": int(len(test_frame)),
        "num_classes": int(frame["label"].nunique()),
        "attributes_csv": str(attributes_path) if attributes_path else None,
    }
    save_json(summary, args.output_dir / "dataset_summary.json")
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
