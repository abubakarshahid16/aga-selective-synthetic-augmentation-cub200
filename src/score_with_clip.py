from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import open_clip
import pandas as pd
import torch
from PIL import Image

from utils import ROOT_DIR, ensure_output_dirs, get_logger, load_class_names


def build_prompts(class_names: List[str]) -> List[str]:
    """Build CLIP text prompts from class names."""

    return [f"a photo of a {name}" for name in class_names]


def main() -> None:
    """Score synthetic candidates using CLIP image-text consistency."""

    parser = argparse.ArgumentParser(description="Score synthetic candidates with CLIP.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates" / "synthetic_manifest.csv")
    parser.add_argument("--class-names", type=Path, default=ROOT_DIR / "data" / "prepared" / "class_names.json")
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_clip_scores.csv")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("score_with_clip", dirs["logs"] / "score_with_clip.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(args.model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize CLIP. Confirm internet/model cache access and the requested OpenCLIP weights."
        ) from exc
    class_names = load_class_names(args.class_names)
    prompts = build_prompts(class_names)
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    frame = pd.read_csv(args.manifest)
    rows = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        try:
            images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch["image_path"].tolist()]).to(device)
        except Exception as exc:
            raise RuntimeError("Failed while loading synthetic images for CLIP scoring.") from exc
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
        for row, scores in zip(batch.to_dict(orient="records"), logits.cpu().tolist()):
            intended = int(row["label"])
            row["clip_intended_score"] = float(scores[intended])
            row["clip_best_label"] = int(torch.tensor(scores).argmax().item())
            row["clip_margin"] = float(scores[intended] - max(score for idx, score in enumerate(scores) if idx != intended))
            rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(args.output_csv, index=False)
    logger.info("Saved CLIP scores to %s", args.output_csv)


if __name__ == "__main__":
    main()
