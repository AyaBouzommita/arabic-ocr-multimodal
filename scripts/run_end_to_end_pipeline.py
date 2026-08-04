"""End-to-End pipeline evaluation: YOLOv11s + PaddleOCR + MARBERT.

This script:
1. Runs the full E2E pipeline (Vision -> OCR -> NLP Correction -> Fusion)
2. Compares CER/WER metrics and saves results.

Usage:
    python -m scripts.run_end_to_end_pipeline
    python -m scripts.run_end_to_end_pipeline --use-gpu
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ocr.paddleocr.engine import PaddleOCREngine
from arabic_ocr_platform.pipeline.pipeline import OCRPipeline
from evaluation.ground_truth import GroundTruthLoader
from evaluation.metrics import compute_cer, compute_wer

YOLO_WEIGHTS = "runs/detect/evaluation/yolo_comparison/yolov11s/weights/best.pt"

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate End-to-End Pipeline (YOLOv11s + PaddleOCR + MARBERT)"
    )
    parser.add_argument("--image-dir", default="data/raw", help="Image directory")
    parser.add_argument("--gt-dir", default="data/ground_truth", help="Ground truth directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--yolo-weights", default=YOLO_WEIGHTS, help="Path to YOLO weights")
    args = parser.parse_args()

    print("=" * 70)
    print("  E2E PIPELINE: YOLOv11s + PaddleOCR + MARBERT")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")

    print(f"\n[Loading] YOLO weights: {args.yolo_weights}")
    try:
        yolo_model = YOLO(args.yolo_weights)
    except Exception as e:
        print(f"Could not load YOLO model: {e}")
        print("Falling back to None (Full image OCR)")
        yolo_model = None
        
    print("[Loading] PaddleOCR engine...")
    try:
        paddle_engine = PaddleOCREngine(lang="ar", use_gpu=args.use_gpu)
    except Exception as e:
        print(f"Could not load PaddleOCR engine: {e}")
        paddle_engine = None

    print("[Loading] E2E Pipeline (including MARBERT)...")
    pipeline = OCRPipeline(vision_model=yolo_model, ocr_model=paddle_engine)

    loader = GroundTruthLoader(image_dir=args.image_dir, gt_dir=args.gt_dir)
    pairs = loader.get_pairs()
    summary = loader.summary()

    print(f"\n[Corpus] {args.image_dir}")
    print(f"   Total images:    {summary['total_images']}")
    print(f"   With ground GT:  {summary['paired']}")

    if summary["paired"] == 0:
        print("\n[ERROR] No image/ground-truth pairs found!")
        sys.exit(1)

    records = []

    print(f"\n[Running] Evaluating {summary['paired']} documents...\n")

    for image_path, gt_text in pairs:
        doc_id = image_path.stem
        print(f"  Processing: {doc_id}...", end=" ")

        try:
            output = pipeline.process(str(image_path), document_id=doc_id)
            ocr_text = output['ocr']['raw_text']
            processing_time_ms = output['ocr']['processing_time_ms']
            tokens = output['ocr']['tokens']
            
            # compute avg confidence
            avg_conf = sum(t['confidence'] for t in tokens) / len(tokens) if tokens else 0.0
            avg_conf = avg_conf * 100 if avg_conf <= 1.0 else avg_conf # scale if it's 0-1
        except Exception as e:
            print(f"Error: {e}")
            continue

        cer = compute_cer(gt_text, ocr_text)
        wer = compute_wer(gt_text, ocr_text)
        
        print(f"CER={cer:.4f}  WER={wer:.4f}  Time={processing_time_ms}ms")

        records.append({
            "document_id": doc_id,
            "method": "E2E Pipeline (YOLO+Paddle+MARBERT)",
            "ground_truth": gt_text,
            "ocr_text": ocr_text,
            "cer": cer,
            "wer": wer,
            "avg_confidence": avg_conf,
            "processing_time_ms": processing_time_ms,
            "token_count": len(tokens),
        })

    df = pd.DataFrame(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sprint3_e2e_pipeline_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    if not df.empty:
        avg_cer = df["cer"].mean()
        avg_wer = df["wer"].mean()
        avg_conf = df["avg_confidence"].mean()
        total_time = df["processing_time_ms"].sum()

        print(f"    Average CER:     {avg_cer:.4f} ({avg_cer*100:.1f}%)")
        print(f"    Average WER:     {avg_wer:.4f} ({avg_wer*100:.1f}%)")
        print(f"    Avg Confidence:  {avg_conf:.1f}%")
        print(f"    Total Time:      {total_time} ms")

    print(f"\n[DONE] Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
