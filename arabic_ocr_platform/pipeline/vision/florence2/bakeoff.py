"""Bake-off report generation for vision detection candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


INTEGRATION_SCORES = {
    "florence2": {
        "score": 3,
        "notes": "Requires transformers + peft + GPU for practical training; strong multi-task flexibility.",
    },
    "yolov8": {
        "score": 5,
        "notes": "Fastest integration for pure detection; minimal dependencies.",
    },
    "detectron2": {
        "score": 2,
        "notes": "Powerful but heavier setup and slower iteration on Windows.",
    },
}


def _load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def generate_florence_candidate_report(metrics: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Florence-2 Candidate Report (Vision Bake-off 3/3)

## Candidate
- Model: Florence-2 (LoRA fine-tuned)
- Dataset: shared team YOLO dataset (`data/vision/detection/`)
- Branch: `florence2-detection-engine`

## Training Summary
- Train samples: {metrics.get('train_samples', 'N/A')}
- Validation samples: {metrics.get('val_samples', 'N/A')}
- Epochs: {metrics.get('epochs', 'N/A')}
- Device: {metrics.get('device', 'N/A')}
- Final train loss: {metrics.get('final_train_loss', 'N/A')}
- Final val loss: {metrics.get('final_val_loss', 'N/A')}

## Evaluation (test split)
- mAP@0.5: **{metrics.get('map50', 'N/A')}**
- mAP@0.5:0.95 (class-avg): **{metrics.get('map50_95', 'N/A')}**
- Avg inference speed: **{metrics.get('avg_inference_ms', 'N/A')} ms/image**
- Evaluated images: {metrics.get('eval_images', 'N/A')}

## Integration Effort
- Score (1=easy, 5=hard): **{INTEGRATION_SCORES['florence2']['score']}/5**
- Notes: {INTEGRATION_SCORES['florence2']['notes']}

## Strengths
- One model can support detection and future multimodal tasks (caption/OCR/grounding).
- Works on the shared annotated dataset without reformatting the whole pipeline.

## Limitations
- Lower mAP than dedicated detectors is common for Florence-2.
- Training and inference are slower without GPU.
- Class vocabulary is open-ended, so post-filtering by allowed classes is required.

## Artifacts
- Model: `{metrics.get('model_dir', 'results/florence2/model')}`
- Metrics: `{metrics.get('metrics_path', 'results/florence2/metrics.json')}`
- Predictions: `{metrics.get('predictions_dir', 'results/florence2/predictions')}`
"""
    output_path.write_text(content, encoding="utf-8")


def generate_bakeoff_comparison(
    florence_metrics: Dict,
    output_path: Path,
    yolo_metrics_path: Optional[Path] = None,
    detectron_metrics_path: Optional[Path] = None,
) -> None:
    yolo = _load_json(yolo_metrics_path) if yolo_metrics_path else None
    detectron = _load_json(detectron_metrics_path) if detectron_metrics_path else None

    def row(name: str, data: Optional[Dict], integration_key: str) -> str:
        if not data:
            return (
                f"| {name} | TBD | TBD | TBD | "
                f"{INTEGRATION_SCORES[integration_key]['score']}/5 | Pending teammate results |"
            )
        return (
            f"| {name} | {data.get('map50', 'TBD')} | {data.get('map50_95', 'TBD')} | "
            f"{data.get('avg_inference_ms', 'TBD')} | "
            f"{INTEGRATION_SCORES[integration_key]['score']}/5 | "
            f"{INTEGRATION_SCORES[integration_key]['notes']} |"
        )

    content = f"""# Vision Detection Bake-off — 3-Way Comparison

Comparison used for the team decision between detection candidates.

| Candidate | mAP@0.5 | mAP@0.5:0.95 | Speed (ms/img) | Integration effort | Notes |
|---|---:|---:|---:|---:|---|
{row('YOLOv8 / PP-YOLOE', yolo, 'yolov8')}
{row('Detectron2', detectron, 'detectron2')}
{row('Florence-2 (ours)', florence_metrics, 'florence2')}

## Decision Criteria
1. **Detection quality**: prioritize higher mAP on the shared test split.
2. **Runtime**: lower ms/image is better for production OCR pipeline latency.
3. **Integration effort**: simpler deployment and maintenance wins when scores are close.

## Current Recommendation
- Use Florence-2 if multimodal flexibility is strategic for later fusion stages.
- Use YOLO/PP-YOLOE if pure detection speed and mAP dominate.
- Use Detectron2 if the team already standardized on COCO/MMDetection tooling.

## Florence-2 Evidence
- Test images evaluated: {florence_metrics.get('eval_images', 'N/A')}
- Metrics file: `results/florence2/metrics.json`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
