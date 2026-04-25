from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils import ROOT_DIR, copy_if_needed, ensure_output_dirs, get_logger, save_image_grid, set_seed


def compute_foreground_mask(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Estimate a foreground mask using GrabCut initialized from the CUB bounding box."""

    h, w = image_bgr.shape[:2]
    x, y, bw, bh = bbox
    rect = (
        max(0, int(x)),
        max(0, int(y)),
        max(1, min(int(bw), w - int(x))),
        max(1, min(int(bh), h - int(y))),
    )
    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, rect, bg_model, fg_model, 4, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    binary = cv2.GaussianBlur(binary, (9, 9), 0)
    return np.clip(binary, 0.0, 1.0)


def build_background_pool(train_frame: pd.DataFrame, image_root: Path) -> List[Path]:
    """Collect real training images that can serve as alternate backgrounds."""

    return [image_root / item for item in train_frame["relative_path"].tolist()]


def sample_background(background_pool: List[Path], output_size: Tuple[int, int], rng: random.Random) -> np.ndarray:
    """Sample a background image and resize it to match the source image."""

    if not background_pool:
        noise = np.random.randint(0, 255, size=(output_size[1], output_size[0], 3), dtype=np.uint8)
        return cv2.GaussianBlur(noise, (21, 21), 0)
    chosen = str(rng.choice(background_pool))
    background = cv2.imread(chosen)
    if background is None:
        noise = np.random.randint(0, 255, size=(output_size[1], output_size[0], 3), dtype=np.uint8)
        return cv2.GaussianBlur(noise, (21, 21), 0)
    return cv2.resize(background, output_size)


def compose_candidate(
    source_bgr: np.ndarray,
    background_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """Create a foreground-preserving synthetic candidate with moderate appearance variation."""

    alpha = np.expand_dims(foreground_mask, axis=-1)
    fg = source_bgr.astype(np.float32)
    bg = background_bgr.astype(np.float32)
    color_scale = np.array(
        [rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08)],
        dtype=np.float32,
    )
    fg = np.clip(fg * color_scale, 0, 255)
    bg = cv2.GaussianBlur(bg, (11, 11), 0)
    mixed = fg * alpha + bg * (1.0 - alpha)
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        mixed = cv2.flip(mixed, 1)
    return mixed


def main() -> None:
    """Generate practical AGA-inspired foreground-preserving synthetic candidates."""

    parser = argparse.ArgumentParser(description="Generate AGA-inspired synthetic bird candidates.")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT_DIR / "data" / "prepared")
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "data" / "CUB_200_2011" / "images")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates")
    parser.add_argument("--samples-per-image", type=int, default=2)
    parser.add_argument("--max-real-images-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-generator-note", type=str, default="")
    parser.add_argument("--external-manifest", type=Path, default=None)
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("generate_aga_style_samples", dirs["logs"] / "generate_aga_style_samples.log")
    set_seed(args.seed)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.external_manifest is not None:
        if not args.external_manifest.exists():
            raise FileNotFoundError(f"External manifest not found: {args.external_manifest}")
        destination = args.output_dir / "synthetic_manifest.csv"
        copy_if_needed(args.external_manifest, destination)
        logger.info("Copied external synthetic manifest to %s", destination)
        return

    train_frame = pd.read_csv(args.prepared_dir / "train.csv")
    background_pool = build_background_pool(train_frame, args.image_root)
    manifest_rows: List[Dict] = []

    for class_id, class_frame in train_frame.groupby("label"):
        class_generated = 0
        sampled_frame = class_frame.sample(n=min(len(class_frame), args.max_real_images_per_class), random_state=args.seed)
        for _, row in sampled_frame.iterrows():
            image_path = args.image_root / row["relative_path"]
            source_bgr = cv2.imread(str(image_path))
            if source_bgr is None:
                logger.warning("Skipping unreadable image: %s", image_path)
                continue
            bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
            mask = compute_foreground_mask(source_bgr, bbox)
            h, w = source_bgr.shape[:2]
            class_dir = args.output_dir / f"class_{int(class_id):03d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(args.samples_per_image):
                background = sample_background(background_pool, (w, h), rng)
                candidate = compose_candidate(source_bgr, background, mask, rng)
                stem = f"img{int(row['image_id']):05d}_aug{sample_index:02d}"
                candidate_path = class_dir / f"{stem}.jpg"
                cv2.imwrite(str(candidate_path), candidate)
                manifest_rows.append(
                    {
                        "image_id": f"synthetic_{int(row['image_id'])}_{sample_index}",
                        "parent_image_id": int(row["image_id"]),
                        "label": int(class_id),
                        "class_name_readable": row["class_name_readable"],
                        "relative_path": str(candidate_path.relative_to(args.output_dir)),
                        "image_path": str(candidate_path.resolve()),
                        "source": "synthetic",
                        "generation_method": "foreground_preserving_background_swap",
                        "external_generator_note": args.external_generator_note,
                    }
                )
                class_generated += 1
        logger.info("Generated %d synthetic samples for class %d", class_generated, int(class_id))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "synthetic_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    save_image_grid(
        [Path(item) for item in manifest["image_path"].head(16).tolist()],
        dirs["samples"] / "synthetic_candidate_grid.png",
        "AGA-Inspired Synthetic Candidates",
    )
    logger.info("Saved %d synthetic candidates to %s", len(manifest), manifest_path)


if __name__ == "__main__":
    main()
