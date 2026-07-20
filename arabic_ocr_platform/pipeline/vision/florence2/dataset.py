"""PyTorch dataset for Florence-2 object-detection fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from arabic_ocr_platform.pipeline.vision.yolo_dataset import export_florence_annotations


class FlorenceODDataset(Dataset):
    """Dataset of prefix/suffix pairs for Florence-2 OD fine-tuning."""

    def __init__(self, annotations_path: Path):
        self.records: List[Dict] = json.loads(annotations_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        return {
            "image": image,
            "prefix": record["prefix"],
            "suffix": record["suffix"],
            "image_path": record["image"],
            "split": record.get("split", "unknown"),
        }


def ensure_florence_annotations(
    output_dir: Path,
    splits: Optional[List[str]] = None,
    limit_per_split: Optional[int] = None,
    dataset_yaml: Optional[Path] = None,
    max_boxes_per_image: Optional[int] = None,
) -> Dict[str, Path]:
    """Create per-split Florence annotation JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for split in splits or ["train", "valid", "test"]:
        path = output_dir / f"{split}.json"
        export_florence_annotations(
            path,
            splits=[split],
            limit_per_split=limit_per_split,
            dataset_yaml=dataset_yaml,
            max_boxes_per_image=max_boxes_per_image,
        )
        split_paths[split] = path
    return split_paths
