from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import ROOT_DIR, ensure_output_dirs, get_logger
from utils import metadata_batch_to_rows


class ManifestImageDataset(Dataset):
    """Dataset for loading images directly from a manifest file."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """Return dataset size."""

        return len(self.frame)

    def __getitem__(self, index: int):
        """Return a transformed image and its metadata row."""

        row = self.frame.iloc[index].to_dict()
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), row


def greedy_prune(group: pd.DataFrame, embeddings: torch.Tensor, similarity_threshold: float) -> pd.DataFrame:
    """Greedily keep diverse samples under a cosine-similarity threshold."""

    kept_indices: List[int] = []
    for idx in range(len(group)):
        if not kept_indices:
            kept_indices.append(idx)
            continue
        similarities = F.cosine_similarity(embeddings[idx].unsqueeze(0), embeddings[kept_indices], dim=1)
        if float(similarities.max().item()) < similarity_threshold:
            kept_indices.append(idx)
    return group.iloc[kept_indices].copy()


def main() -> None:
    """Extract DINOv2 embeddings and prune near-duplicate synthetic samples."""

    parser = argparse.ArgumentParser(description="Prune synthetic candidates with DINOv2 diversity filtering.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "outputs" / "samples" / "synthetic_candidates" / "synthetic_manifest.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.96)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_dino_pruned.csv")
    parser.add_argument("--embeddings-csv", type=Path, default=ROOT_DIR / "outputs" / "tables" / "synthetic_dino_embeddings.csv")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("prune_with_dino", dirs["logs"] / "prune_with_dino.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load DINOv2 through torch.hub. Confirm internet/model cache access before rerunning."
        ) from exc
    model.eval()

    frame = pd.read_csv(args.manifest)
    dataset = ManifestImageDataset(frame)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    embeddings: List[torch.Tensor] = []
    rows: List[Dict] = []
    offset = 0
    with torch.no_grad():
        for images, metadata in loader:
            images = images.to(device, non_blocking=True)
            features = model(images)
            features = F.normalize(features, dim=1).cpu()
            embeddings.append(features)
            for row, embedding in zip(metadata_batch_to_rows(metadata), features):
                record = dict(row)
                record["embedding"] = json_dumps_tensor(embedding)
                rows.append(record)

    embedding_tensor = torch.cat(embeddings, dim=0)
    pd.DataFrame(rows).to_csv(args.embeddings_csv, index=False)

    kept_groups = []
    per_class_stats: List[Dict[str, object]] = []
    for label, group in frame.groupby("label", sort=True):
        group_indices = group.index.to_list()
        group_embeddings = embedding_tensor[group_indices]
        pruned = greedy_prune(group.reset_index(drop=True), group_embeddings, args.similarity_threshold)
        kept_groups.append(pruned)
        per_class_stats.append(
            {
                "label": int(label),
                "num_candidates": int(len(group)),
                "num_kept": int(len(pruned)),
                "num_removed": int(len(group) - len(pruned)),
                "keep_ratio": float(len(pruned) / max(1, len(group))),
            }
        )
    kept_frame = pd.concat(kept_groups, ignore_index=True)
    kept_frame.to_csv(args.output_csv, index=False)
    stats = pd.DataFrame(
        [
            {
                "num_candidates": int(len(frame)),
                "num_kept": int(len(kept_frame)),
                "num_removed": int(len(frame) - len(kept_frame)),
                "keep_ratio": float(len(kept_frame) / max(1, len(frame))),
                "similarity_threshold": args.similarity_threshold,
            }
        ]
    )
    stats.to_csv(dirs["tables"] / "dino_pruning_stats.csv", index=False)
    pd.DataFrame(per_class_stats).to_csv(dirs["tables"] / "dino_pruning_per_class_stats.csv", index=False)
    logger.info("Kept %d / %d synthetic samples after DINO pruning", len(kept_frame), len(frame))


def json_dumps_tensor(tensor: torch.Tensor) -> str:
    """Serialize a tensor to a compact JSON string."""

    return "[" + ",".join(f"{float(x):.6f}" for x in tensor.tolist()) + "]"


if __name__ == "__main__":
    main()
