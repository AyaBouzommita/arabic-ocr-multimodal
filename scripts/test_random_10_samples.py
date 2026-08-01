"""Test Fine-Tuned YOLOv11s + PaddleOCR on 10 random images from the new dataset."""

import random
import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ocr.paddleocr.engine import PaddleOCREngine
from scripts.run_yolo_paddleocr_pipeline import run_paddleocr_alone, run_yolo_paddleocr_pipeline
from evaluation.metrics import compute_cer, compute_wer

YOLO_WEIGHTS = "runs/detect/evaluation/yolo_comparison/yolov11s_finetuned/weights/best.pt"
VAL_IMG_DIR = Path("data/finetune_dataset/images/val")

random.seed(42)

def main():
    print("=" * 70)
    print("  TESTING FINE-TUNED PIPELINE ON 10 RANDOM SAMPLES FROM NEW DATASET")
    print("=" * 70)

    # Get all images in val directory
    images = list(VAL_IMG_DIR.glob("*.png")) + list(VAL_IMG_DIR.glob("*.jpg"))
    if not images:
        print("No images found in data/finetune_dataset/images/val!")
        return

    # Select 10 random images
    selected_images = random.sample(images, min(10, len(images)))

    print(f"\n[Loading] Fine-Tuned YOLOv11s from: {YOLO_WEIGHTS}")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("[Loading] PaddleOCR engine...")
    engine = PaddleOCREngine(lang="ar", use_gpu=torch.cuda.is_available())

    print(f"\nEvaluating {len(selected_images)} random samples...\n")

    records = []

    for i, img_path in enumerate(selected_images, 1):
        base_name = img_path.stem
        # Look for matching .gt.txt in OCR_GS_Data
        gt_files = list(Path("data/OCR_GS_Data/ara").rglob(f"{base_name}.gt.txt"))
        gt_text = ""
        if gt_files:
            gt_text = gt_files[0].read_text(encoding="utf-8").strip()

        # Run PaddleOCR Alone
        res_alone = run_paddleocr_alone(engine, img_path, base_name)
        cer_alone = compute_cer(gt_text, res_alone.raw_text)
        wer_alone = compute_wer(gt_text, res_alone.raw_text)

        # Run Fine-Tuned YOLOv11s + PaddleOCR
        res_yolo = run_yolo_paddleocr_pipeline(yolo_model, engine, img_path, base_name, conf_thresh=0.25)
        cer_yolo = compute_cer(gt_text, res_yolo.raw_text)
        wer_yolo = compute_wer(gt_text, res_yolo.raw_text)

        records.append({
            "idx": i,
            "id": base_name[:8],
            "gt": gt_text,
            "alone_text": res_alone.raw_text,
            "yolo_text": res_yolo.raw_text,
            "cer_alone": cer_alone,
            "cer_yolo": cer_yolo,
            "wer_alone": wer_alone,
            "wer_yolo": wer_yolo,
        })

        print(f"[{i}/10] Image: {base_name[:12]}")
        print(f"  Ground Truth:  {gt_text}")
        print(f"  Paddle Alone:  {res_alone.raw_text}  (CER: {cer_alone:.4f})")
        print(f"  YOLO+Paddle:   {res_yolo.raw_text}  (CER: {cer_yolo:.4f})")
        print("-" * 70)

    # Print Summary Table
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (10 Random New Dataset Samples)")
    print("=" * 70)
    print(f"{'Idx':<4} {'Image ID':<10} {'Alone CER':>10} {'YOLO CER':>10} {'Alone WER':>10} {'YOLO WER':>10}")
    print("-" * 70)
    for r in records:
        print(f"{r['idx']:<4} {r['id']:<10} {r['cer_alone']:>10.4f} {r['cer_yolo']:>10.4f} {r['wer_alone']:>10.4f} {r['wer_yolo']:>10.4f}")

    avg_cer_alone = sum(r["cer_alone"] for r in records) / len(records)
    avg_cer_yolo = sum(r["cer_yolo"] for r in records) / len(records)
    avg_wer_alone = sum(r["wer_alone"] for r in records) / len(records)
    avg_wer_yolo = sum(r["wer_yolo"] for r in records) / len(records)

    print("\n" + "=" * 70)
    print("  AVERAGES ON NEW DATASET SAMPLES")
    print("=" * 70)
    print(f"  PaddleOCR Alone CER:  {avg_cer_alone:.4f}")
    print(f"  YOLO+PaddleOCR CER:   {avg_cer_yolo:.4f}")
    print(f"  PaddleOCR Alone WER:  {avg_wer_alone:.4f}")
    print(f"  YOLO+PaddleOCR WER:   {avg_wer_yolo:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
