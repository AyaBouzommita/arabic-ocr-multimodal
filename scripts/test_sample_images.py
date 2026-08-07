"""Test Fine-Tuned YOLOv11s + PaddleOCR specifically on sample.png and sample2.png."""

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

def test_image(yolo_model, engine, image_name):
    img_path = Path("data/raw") / f"{image_name}.png"
    gt_path = Path("data/ground_truth") / f"{image_name}.txt"

    if not img_path.exists():
        print(f"Error: {img_path} not found!")
        return

    gt_text = gt_path.read_text(encoding="utf-8").strip() if gt_path.exists() else ""

    print("=" * 70)
    print(f"  TESTING IMAGE: {image_name}.png")
    print("=" * 70)
    print(f"  Ground Truth: {gt_text}\n")

    # 1. PaddleOCR Alone
    res_alone = run_paddleocr_alone(engine, img_path, image_name)
    cer_alone = compute_cer(gt_text, res_alone.raw_text)
    wer_alone = compute_wer(gt_text, res_alone.raw_text)

    print("--- [1] PaddleOCR Alone ---")
    print(f"  Extracted Text: {res_alone.raw_text}")
    print(f"  Confidence:     {res_alone.avg_confidence:.1f}%")
    print(f"  Processing Time: {res_alone.processing_time_ms} ms")
    print(f"  CER: {cer_alone:.4f}  |  WER: {wer_alone:.4f}")

    # 2. YOLOv11s Fine-Tuned + PaddleOCR
    res_yolo = run_yolo_paddleocr_pipeline(yolo_model, engine, img_path, image_name, conf_thresh=0.25)
    cer_yolo = compute_cer(gt_text, res_yolo.raw_text)
    wer_yolo = compute_wer(gt_text, res_yolo.raw_text)

    print("\n--- [2] Fine-Tuned YOLOv11s + PaddleOCR ---")
    print(f"  Extracted Text: {res_yolo.raw_text}")
    print(f"  Confidence:     {res_yolo.avg_confidence:.1f}%")
    print(f"  Processing Time: {res_yolo.processing_time_ms} ms")
    print(f"  CER: {cer_yolo:.4f}  |  WER: {wer_yolo:.4f}")

    print("\n" + "-" * 70)

def main():
    print(f"Loading Fine-Tuned YOLOv11s from: {YOLO_WEIGHTS}")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("Loading PaddleOCR engine...")
    engine = PaddleOCREngine(lang="ar", use_gpu=torch.cuda.is_available())

    test_image(yolo_model, engine, "sample")
    test_image(yolo_model, engine, "sample2")

if __name__ == "__main__":
    main()
