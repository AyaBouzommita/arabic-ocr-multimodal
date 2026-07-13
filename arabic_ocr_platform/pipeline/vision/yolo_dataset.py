"""Utilities for loading the team YOLO object-detection dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image


@dataclass
class YoloBox:
    """Single YOLO annotation."""

    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy_pixels(self, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """Convert normalized YOLO box to pixel xyxy."""
        w = self.width * image_width
        h = self.height * image_height
        x_c = self.x_center * image_width
        y_c = self.y_center * image_height
        x1 = max(0.0, x_c - w / 2)
        y1 = max(0.0, y_c - h / 2)
        x2 = min(float(image_width), x_c + w / 2)
        y2 = min(float(image_height), y_c + h / 2)
        return x1, y1, x2, y2


@dataclass
class YoloSample:
    """Image path with parsed YOLO boxes."""

    image_path: Path
    split: str
    boxes: List[YoloBox]
    image_width: int
    image_height: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_dataset_root(dataset_yaml: Optional[Path] = None) -> Path:
    """Resolve dataset root from data/vision/dataset.yaml."""
    yaml_path = dataset_yaml or (project_root() / "data" / "vision" / "dataset.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base = project_root() / "data" / "vision"
    path_value = config.get("path", "detection")
    dataset_root = Path(path_value)
    if not dataset_root.is_absolute():
        dataset_root = (base / dataset_root).resolve()
    return dataset_root


def load_dataset_config(dataset_yaml: Optional[Path] = None) -> Dict:
    yaml_path = dataset_yaml or (project_root() / "data" / "vision" / "dataset.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names_from_config(config: Dict) -> Dict[int, str]:
    names = config.get("names", {})
    return {int(k): str(v) for k, v in names.items()}


def split_dirs(dataset_root: Path, config: Dict) -> Dict[str, Path]:
    return {
        "train": dataset_root / Path(config["train"]).parent.parent.name,
        "valid": dataset_root / Path(config["val"]).parent.parent.name,
        "test": dataset_root / Path(config["test"]).parent.parent.name,
    }


def list_split_images(split_dir: Path) -> List[Path]:
    image_dir = split_dir / "images"
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )


def parse_label_file(label_path: Path, class_map: Dict[int, str]) -> List[YoloBox]:
    if not label_path.exists():
        return []

    boxes: List[YoloBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        boxes.append(
            YoloBox(
                class_id=class_id,
                class_name=class_map.get(class_id, str(class_id)),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def load_yolo_sample(
    image_path: Path,
    split: str,
    class_map: Dict[int, str],
) -> YoloSample:
    with Image.open(image_path) as img:
        width, height = img.size

    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    boxes = parse_label_file(label_path, class_map)
    return YoloSample(
        image_path=image_path,
        split=split,
        boxes=boxes,
        image_width=width,
        image_height=height,
    )


def load_split_samples(
    split: str,
    limit: Optional[int] = None,
    dataset_yaml: Optional[Path] = None,
) -> List[YoloSample]:
    config = load_dataset_config(dataset_yaml)
    dataset_root = resolve_dataset_root(dataset_yaml)
    class_map = class_names_from_config(config)
    split_map = split_dirs(dataset_root, config)
    split_dir = split_map[split]

    samples = []
    for image_path in list_split_images(split_dir):
        samples.append(load_yolo_sample(image_path, split, class_map))
        if limit and len(samples) >= limit:
            break
    return samples


def export_florence_annotations(
    output_path: Path,
    splits: Optional[List[str]] = None,
    limit_per_split: Optional[int] = None,
    dataset_yaml: Optional[Path] = None,
) -> Path:
    """Export Florence-2 OD JSON annotations for selected splits."""
    splits = splits or ["train", "valid", "test"]
    records = []

    for split in splits:
        for sample in load_split_samples(split, limit=limit_per_split, dataset_yaml=dataset_yaml):
            suffix_parts = []
            for box in sample.boxes:
                x1, y1, x2, y2 = box.to_xyxy_pixels(sample.image_width, sample.image_height)
                qx1 = int(round((x1 / sample.image_width) * 1000))
                qy1 = int(round((y1 / sample.image_height) * 1000))
                qx2 = int(round((x2 / sample.image_width) * 1000))
                qy2 = int(round((y2 / sample.image_height) * 1000))
                suffix_parts.append(
                    f"{box.class_name}<loc_{qx1}><loc_{qy1}><loc_{qx2}><loc_{qy2}>"
                )
            records.append(
                {
                    "prefix": "<OD>",
                    "suffix": "".join(suffix_parts),
                    "image": str(sample.image_path),
                    "split": split,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
