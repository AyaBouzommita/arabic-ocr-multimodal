"""Full Integrated OCR Pipeline with Smart Confidence-Gated AraBERT Correction.

Pipeline flow:
1. YOLOv11s Fine-Tuned (Layout Detection & Cropping)
2. PaddleOCR (Text Recognition with Confidence Scores)
3. RTL Direction Corrector (Natural Arabic connected text order)
4. Smart AraBERT Masked LM (Strict 1-char Typo Correction ONLY for tokens with confidence < 80%)
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from ultralytics import YOLO

from ocr.paddleocr.engine import PaddleOCREngine
from ocr.postprocessing.arabert_corrector import AraBERTCorrector
from scripts.run_yolo_paddleocr_pipeline import run_yolo_paddleocr_pipeline, run_paddleocr_alone, fix_arabic_direction
from evaluation.ground_truth import GroundTruthLoader
from evaluation.metrics import compute_cer, compute_wer

YOLO_WEIGHTS = "runs/detect/evaluation/yolo_comparison/yolov11s_finetuned/weights/best.pt"

def main():
    print("=" * 80)
    print("  SMART OCR PIPELINE: YOLOv11s + PaddleOCR + Confidence-Gated AraBERT")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/3] Loading Fine-Tuned YOLOv11s ({YOLO_WEIGHTS})...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("[2/3] Loading PaddleOCR engine...")
    paddle_engine = PaddleOCREngine(lang="ar", use_gpu=torch.cuda.is_available())

    print("[3/3] Loading Smart AraBERT Masked LM...")
    arabert_corrector = AraBERTCorrector(device=device)

    loader = GroundTruthLoader(image_dir="data/raw", gt_dir="data/ground_truth")
    pairs = loader.get_pairs()
    print(f"\n[Dataset] Loaded {len(pairs)} test image/ground-truth pairs.")

    records = []

    print("\n[Running Smart Pipeline Evaluation]...\n")

    for img_path, gt_text in pairs:
        doc_id = img_path.stem

        # Stage 1: PaddleOCR Alone
        res_paddle = run_paddleocr_alone(paddle_engine, img_path, doc_id)
        cer_paddle = compute_cer(gt_text, res_paddle.raw_text)
        wer_paddle = compute_wer(gt_text, res_paddle.raw_text)

        # Stage 2: YOLOv11s + PaddleOCR
        res_yolo = run_yolo_paddleocr_pipeline(yolo_model, paddle_engine, img_path, doc_id, conf_thresh=0.25)
        cer_yolo = compute_cer(gt_text, res_yolo.raw_text)
        wer_yolo = compute_wer(gt_text, res_yolo.raw_text)

        # Stage 3: Smart Confidence-Gated AraBERT Correction
        start_t = time.perf_counter()
        if res_yolo.tokens:
            arabert_text = arabert_corrector.correct_tokens(res_yolo.tokens, conf_threshold=80.0, max_edit_dist=1)
        arabert_text = fix_arabic_direction(arabert_text)
        
        total_time_arabert = res_yolo.processing_time_ms + int((time.perf_counter() - start_t) * 1000)

        cer_arabert = compute_cer(gt_text, arabert_text)
        wer_arabert = compute_wer(gt_text, arabert_text)

        records.append({
            "document_id": doc_id,
            "ground_truth": gt_text,
            "paddle_text": res_paddle.raw_text,
            "yolo_text": res_yolo.raw_text,
            "arabert_text": arabert_text,
            "cer_paddle": cer_paddle,
            "cer_yolo": cer_yolo,
            "cer_arabert": cer_arabert,
            "wer_paddle": wer_paddle,
            "wer_yolo": wer_yolo,
            "wer_arabert": wer_arabert,
            "time_paddle": res_paddle.processing_time_ms,
            "time_yolo": res_yolo.processing_time_ms,
            "time_arabert": total_time_arabert,
        })

        print(f"Document: {doc_id}")
        print(f"  Ground Truth:         {gt_text}")
        print(f"  [1] PaddleOCR Alone:  {res_paddle.raw_text} (CER: {cer_paddle:.4f})")
        print(f"  [2] YOLO + Paddle:    {res_yolo.raw_text} (CER: {cer_yolo:.4f})")
        print(f"  [3] + Smart AraBERT:  {arabert_text} (CER: {cer_arabert:.4f})")
        print("-" * 80)

    df = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print("  SMART PIPELINE EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n  [Stage 1: PaddleOCR Alone]")
    print(f"    Average CER:     {df['cer_paddle'].mean():.4f}")
    print(f"    Average WER:     {df['wer_paddle'].mean():.4f}")
    print(f"    Total Time:      {df['time_paddle'].sum()} ms")

    print(f"\n  [Stage 2: YOLOv11s + PaddleOCR]")
    print(f"    Average CER:     {df['cer_yolo'].mean():.4f}")
    print(f"    Average WER:     {df['wer_yolo'].mean():.4f}")
    print(f"    Total Time:      {df['time_yolo'].sum()} ms")

    print(f"\n  [Stage 3: YOLOv11s + PaddleOCR + Smart AraBERT ★]")
    print(f"    Average CER:     {df['cer_arabert'].mean():.4f}")
    print(f"    Average WER:     {df['wer_arabert'].mean():.4f}")
    print(f"    Total Time:      {df['time_arabert'].sum()} ms")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
