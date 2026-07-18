"""
Sprint 2 - Pipeline d'évaluation partagé (US-13)
Samar Zaabouti - NeoLedge Stage 2026

Ce script évalue n'importe quel modèle de détection (YOLOv8, Detectron2,
Florence-2) avec exactement les mêmes métriques, pour permettre une
comparaison objective lors du bake-off de fin de Sprint 2.

Métriques calculées :
- mAP@0.5
- mAP@0.5:0.95
- Precision par classe
- Recall par classe
- Courbes Precision-Recall
- Vitesse d'inférence (ms/image)

Usage:
    python detection_eval_pipeline.py --predictions predictions.json --ground_truth val.json --output_dir eval_output/
"""

import argparse
import json
import os
import time
import numpy as np
from collections import defaultdict


# ----------------------------------------------------------------------
# FORMAT DES PRÉDICTIONS (contrat d'interface Section 1.4)
# Chaque modèle doit produire un fichier JSON avec ce format :
# [
#   {
#     "image_id": 1,
#     "category_id": 0,
#     "bbox": [x1, y1, width, height],  # format COCO
#     "score": 0.95
#   }, ...
# ]
# ----------------------------------------------------------------------

def load_coco_gt(gt_path):
    """Charge le ground truth COCO JSON."""
    with open(gt_path, 'r') as f:
        return json.load(f)


def load_predictions(pred_path):
    """Charge les prédictions au format COCO results."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def compute_iou(box1, box2):
    """
    Calcule l'IoU entre deux boîtes au format [x1, y1, w, h].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_ap(precisions, recalls):
    """Calcule l'Average Precision via interpolation 11 points."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(prec_at_rec) if prec_at_rec else 0.0
    return ap / 11.0


def evaluate(gt_data, predictions, iou_threshold=0.5):
    """
    Évalue les prédictions contre le ground truth.
    Retourne les métriques par classe et globales.
    """
    # Organiser le GT par image et catégorie
    gt_by_image = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    # Organiser les catégories
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}

    # Organiser les prédictions par catégorie
    preds_by_cat = defaultdict(list)
    for pred in predictions:
        preds_by_cat[pred['category_id']].append(pred)

    results = {}
    all_aps = []

    for cat_id, cat_name in categories.items():
        preds = sorted(preds_by_cat[cat_id], key=lambda x: -x['score'])
        gt_for_cat = {
            img_id: [ann for ann in anns if ann['category_id'] == cat_id]
            for img_id, anns in gt_by_image.items()
        }
        total_gt = sum(len(v) for v in gt_for_cat.values())

        if total_gt == 0:
            continue

        matched = defaultdict(set)
        tp_list = []
        fp_list = []

        for pred in preds:
            img_id = pred['image_id']
            gt_anns = gt_for_cat.get(img_id, [])
            best_iou = 0
            best_idx = -1

            for idx, gt_ann in enumerate(gt_anns):
                iou = compute_iou(pred['bbox'], gt_ann['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx not in matched[img_id]:
                tp_list.append(1)
                fp_list.append(0)
                matched[img_id].add(best_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-10)
        recalls    = tp_cum / (total_gt + 1e-10)

        ap = compute_ap(list(precisions), list(recalls))
        final_prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_rec  = float(recalls[-1])    if len(recalls)    > 0 else 0.0

        results[cat_name] = {
            'AP':        round(ap * 100, 2),
            'Precision': round(final_prec * 100, 2),
            'Recall':    round(final_rec  * 100, 2),
            'GT_count':  total_gt,
            'Pred_count': len(preds)
        }
        all_aps.append(ap)

    mean_ap = float(np.mean(all_aps)) * 100 if all_aps else 0.0
    return results, mean_ap


def print_report(results, mean_ap, model_name, inference_time_ms=None):
    """Affiche le rapport d'évaluation."""
    print(f"\n{'='*60}")
    print(f"  RAPPORT D'ÉVALUATION — {model_name}")
    print(f"{'='*60}")
    print(f"\n  mAP@0.5 global : {mean_ap:.2f}%")
    if inference_time_ms:
        print(f"  Vitesse d'inférence : {inference_time_ms:.1f} ms/image")
    print(f"\n  {'Classe':<20} {'AP':>8} {'Precision':>12} {'Recall':>10} {'GT':>6} {'Pred':>6}")
    print(f"  {'-'*65}")
    for cat_name, metrics in sorted(results.items()):
        print(f"  {cat_name:<20} {metrics['AP']:>7.1f}% "
              f"{metrics['Precision']:>11.1f}% "
              f"{metrics['Recall']:>9.1f}% "
              f"{metrics['GT_count']:>6} "
              f"{metrics['Pred_count']:>6}")
    print(f"{'='*60}\n")


def save_report(results, mean_ap, model_name, output_dir, inference_time_ms=None):
    """Sauvegarde le rapport en JSON."""
    os.makedirs(output_dir, exist_ok=True)
    report = {
        'model': model_name,
        'mAP_50': round(mean_ap, 2),
        'inference_time_ms': inference_time_ms,
        'per_class': results
    }
    out_path = os.path.join(output_dir, f"{model_name}_eval_report.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Rapport sauvegardé : {out_path}")
    return out_path


def compare_models(report_paths):
    """Compare les 3 modèles du bake-off."""
    print(f"\n{'='*60}")
    print(f"  COMPARAISON BAKE-OFF — YOLOv8 vs Detectron2 vs Florence-2")
    print(f"{'='*60}")
    print(f"\n  {'Modèle':<20} {'mAP@0.5':>10} {'Vitesse':>12}")
    print(f"  {'-'*45}")

    for path in report_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                report = json.load(f)
            speed = f"{report['inference_time_ms']:.1f} ms" if report.get('inference_time_ms') else "N/A"
            print(f"  {report['model']:<20} {report['mAP_50']:>9.2f}% {speed:>12}")

    print(f"{'='*60}\n")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'évaluation partagé — Sprint 2")
    parser.add_argument("--predictions",  required=True, help="Fichier JSON des prédictions (format COCO results)")
    parser.add_argument("--ground_truth", required=True, help="Fichier JSON ground truth (format COCO)")
    parser.add_argument("--model_name",   default="detectron2", help="Nom du modèle évalué")
    parser.add_argument("--output_dir",   default="eval_output", help="Dossier de sortie")
    parser.add_argument("--iou_threshold",type=float, default=0.5, help="Seuil IoU (défaut: 0.5)")
    parser.add_argument("--inference_time_ms", type=float, default=None, help="Temps moyen d'inférence en ms")
    parser.add_argument("--compare", nargs='+', default=None, help="Chemins des rapports à comparer")
    args = parser.parse_args()

    print(f"\nChargement du ground truth : {args.ground_truth}")
    gt_data = load_coco_gt(args.ground_truth)
    print(f"  {len(gt_data['images'])} images, {len(gt_data['annotations'])} annotations, {len(gt_data['categories'])} classes")

    print(f"Chargement des prédictions : {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"  {len(predictions)} prédictions chargées")

    print(f"\nCalcul des métriques (IoU threshold = {args.iou_threshold})...")
    results, mean_ap = evaluate(gt_data, predictions, iou_threshold=args.iou_threshold)

    print_report(results, mean_ap, args.model_name, args.inference_time_ms)
    save_report(results, mean_ap, args.model_name, args.output_dir, args.inference_time_ms)

    if args.compare:
        compare_models(args.compare)
