"""Detection metrics: IoU, AP, mAP, and speed benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Detection:
    label: str
    bbox: List[float]
    confidence: float = 1.0


def average_precision(
    predictions: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP for one class at a given IoU threshold."""
    if not ground_truth:
        return 0.0 if predictions else 1.0

    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    matched = [False] * len(ground_truth)
    tp, fp = [], []

    for pred in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truth):
            if matched[idx]:
                continue
            iou = box_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for t, f in zip(tp, fp):
        tp_cum += t
        fp_cum += f
        precision = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) else 0.0
        recall = tp_cum / len(ground_truth)
        precisions.append(precision)
        recalls.append(recall)

    if not recalls:
        return 0.0

    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p = max((prec for prec, rec in zip(precisions, recalls) if rec >= t), default=0.0)
        ap += p / 11
    return ap


def compute_precision_recall(
    pred_by_image: Dict[str, List[Detection]],
    gt_by_image: Dict[str, List[Detection]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute global precision and recall at a given IoU threshold."""
    tp = fp = fn = 0

    for image_id, ground_truth in gt_by_image.items():
        predictions = sorted(
            pred_by_image.get(image_id, []),
            key=lambda p: p.confidence,
            reverse=True,
        )
        matched = [False] * len(ground_truth)

        for pred in predictions:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(ground_truth):
                if matched[idx] or pred.label != gt.label:
                    continue
                iou = box_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched[best_idx] = True
                tp += 1
            else:
                fp += 1

        fn += sum(1 for m in matched if not m)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_map(
    pred_by_image: Dict[str, List[Detection]],
    gt_by_image: Dict[str, List[Detection]],
    class_names: Sequence[str],
    iou_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute per-class AP and mean AP."""
    iou_thresholds = iou_thresholds or [0.5]
    per_class = {}
    for class_name in class_names:
        aps = []
        for iou_thr in iou_thresholds:
            class_preds = []
            class_gts = []
            for image_id in gt_by_image:
                class_preds.extend(
                    [p for p in pred_by_image.get(image_id, []) if p.label == class_name]
                )
                class_gts.extend(
                    [g for g in gt_by_image.get(image_id, []) if g.label == class_name]
                )
            aps.append(average_precision(class_preds, class_gts, iou_thr))
        per_class[class_name] = sum(aps) / len(aps)

    map50 = round(sum(per_class.values()) / len(per_class), 4) if per_class else 0.0
    map50_95_values = []
    for iou_thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        class_aps = []
        for class_name in class_names:
            class_preds = [
                p for preds in pred_by_image.values() for p in preds if p.label == class_name
            ]
            class_gts = [
                g for gts in gt_by_image.values() for g in gts if g.label == class_name
            ]
            class_aps.append(average_precision(class_preds, class_gts, iou_thr))
        if class_aps:
            map50_95_values.append(sum(class_aps) / len(class_aps))
    map50_95 = round(sum(map50_95_values) / len(map50_95_values), 4) if map50_95_values else map50

    return {
        **{f"ap_{k}": round(v, 4) for k, v in per_class.items()},
        "map50": map50,
        "map50_95": map50_95,
    }


def save_metrics(path: Path, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
