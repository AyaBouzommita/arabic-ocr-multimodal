"""Evaluate Florence-2 detection: mAP and inference speed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.engine import Florence2Detector
from arabic_ocr_platform.pipeline.vision.florence2.metrics import (
    Detection,
    compute_map,
    compute_precision_recall,
    save_metrics,
)
from arabic_ocr_platform.pipeline.vision.yolo_dataset import class_names_from_config, load_dataset_config, load_split_samples


def evaluate_florence2(
    model_dir: Optional[Path] = None,
    split: str = "test",
    limit: Optional[int] = None,
    config: Optional[Florence2Config] = None,
) -> Dict:
    cfg = config or Florence2Config()
    detector = Florence2Detector(config=cfg, model_dir=model_dir)
    dataset_cfg = load_dataset_config(cfg.dataset_yaml)
    class_map = class_names_from_config(dataset_cfg)
    allowed_classes: Set[str] = set(class_map.values())

    samples = load_split_samples(split, limit=limit, dataset_yaml=cfg.dataset_yaml)
    pred_by_image: Dict[str, List[Detection]] = {}
    gt_by_image: Dict[str, List[Detection]] = {}
    timings: List[int] = []
    predictions_out = []

    for sample in samples:
        image_id = sample.image_path.stem
        gt_by_image[image_id] = [
            Detection(
                label=box.class_name,
                bbox=list(box.to_xyxy_pixels(sample.image_width, sample.image_height)),
                confidence=1.0,
            )
            for box in sample.boxes
        ]

        result = detector.predict_path(sample.image_path, allowed_classes=allowed_classes)
        timings.append(result["processing_time_ms"])
        pred_by_image[image_id] = [
            Detection(label=obj["label"], bbox=obj["bbox"], confidence=obj["confidence"])
            for obj in result["objects"]
        ]
        predictions_out.append(result)

    map_metrics = compute_map(
        pred_by_image,
        gt_by_image,
        class_names=sorted(allowed_classes),
        iou_thresholds=[0.5],
    )
    pr_metrics = compute_precision_recall(pred_by_image, gt_by_image, iou_threshold=0.5)

    metrics = {
        "model": "florence2",
        "candidate": "florence2",
        "split": split,
        "eval_images": len(samples),
        "precision": pr_metrics["precision"],
        "recall": pr_metrics["recall"],
        "map50": map_metrics["map50"],
        "map50_95": map_metrics["map50_95"],
        "avg_inference_ms": round(sum(timings) / len(timings), 2) if timings else 0.0,
        "per_class_ap": {k: v for k, v in map_metrics.items() if k.startswith("ap_")},
        "model_dir": str(model_dir or cfg.checkpoint),
        "metrics_path": str(cfg.metrics_path),
        "predictions_dir": str(cfg.predictions_dir),
        "dataset": str(cfg.dataset_yaml),
    }

    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(cfg.metrics_path, metrics)

    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
    for pred in predictions_out:
        out_file = cfg.predictions_dir / f"{pred['document_id']}.json"
        out_file.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics
