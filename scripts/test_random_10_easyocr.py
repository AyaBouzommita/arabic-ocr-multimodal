"""Test Full 3-Stage Pipeline (PaddleOCR vs YOLO+Paddle vs YOLO+Paddle+AraBERT) on 10 random images."""

import random
import sys
import time
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ocr.easyocr.engine import EasyOCREngine
from ocr.preprocessing import ImageEnhancer
from ocr.postprocessing.arabert_corrector import AraBERTCorrector
from ocr.postprocessing.bilingual_corrector import BilingualCorrector
from scripts.run_yolo_easyocr_pipeline import run_easyocr_alone, run_yolo_easyocr_pipeline, fix_arabic_direction
from evaluation.metrics import compute_cer, compute_wer

YOLO_WEIGHTS = "runs/detect/evaluation/yolo_comparison/yolov11s_finetuned/weights/best.pt"
VAL_IMG_DIR = Path("data/finetune_dataset/images/val")

random.seed()  # Use system time for truly random new images

def main():
    print("=" * 80)
    print("  3-STAGE PIPELINE EVALUATION ON 10 RANDOM IMAGES")
    print("  [1] PaddleOCR Alone  vs  [2] YOLO+PaddleOCR  vs  [3] YOLO+PaddleOCR+AraBERT")
    print("=" * 80)

    images = list(VAL_IMG_DIR.glob("*.png")) + list(VAL_IMG_DIR.glob("*.jpg"))
    if not images:
        print("No images found in data/finetune_dataset/images/val!")
        return

    selected_images = random.sample(images, min(10, len(images)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Loading] Fine-Tuned YOLOv11s from: {YOLO_WEIGHTS}")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("[Loading] EasyOCR engine (ar, en)...")
    easyocr_engine = EasyOCREngine(languages=["ar", "en"], gpu=torch.cuda.is_available())

    print("[Loading] Bilingual Corrector...")
    arabert_corrector = BilingualCorrector(device=device)

    print("[Loading] Image Enhancer (CLAHE + Denoise)...")
    enhancer = ImageEnhancer()

    print(f"\nEvaluating {len(selected_images)} random samples...\n")

    records = []

    for i, img_path in enumerate(selected_images, 1):
        base_name = img_path.stem
        gt_files = list(Path("data/OCR_GS_Data/ara").rglob(f"{base_name}.gt.txt"))
        gt_text = ""
        if gt_files:
            gt_text = gt_files[0].read_text(encoding="utf-8").strip()
            # GT in file is stored backwards. We MUST fix it to compare with forward predictions.
            gt_text = fix_arabic_direction(gt_text)

        # Method 1: EasyOCR Alone
        res_alone = run_easyocr_alone(easyocr_engine, img_path, base_name)
        cer_alone = compute_cer(gt_text, res_alone.raw_text)
        wer_alone = compute_wer(gt_text, res_alone.raw_text)

        # Method 2: Fine-Tuned YOLO + EasyOCR + Enhancement
        res_yolo = run_yolo_easyocr_pipeline(yolo_model, easyocr_engine, img_path, base_name, conf_thresh=0.25, enhancer=enhancer)
        cer_yolo = compute_cer(gt_text, res_yolo.raw_text)
        wer_yolo = compute_wer(gt_text, res_yolo.raw_text)

        # Method 3: YOLO + EasyOCR + AraBERT
        start_t = time.perf_counter()
        arabert_text = arabert_corrector.correct_tokens(res_yolo.tokens, max_edit_dist=2, conf_threshold=90.0)
        cer_arabert = compute_cer(gt_text, arabert_text)
        wer_arabert = compute_wer(gt_text, arabert_text)

        records.append({
            "idx": i,
            "id": base_name[:8],
            "gt": gt_text,
            "alone_text": res_alone.raw_text,
            "yolo_text": res_yolo.raw_text,
            "arabert_text": arabert_text,
            "cer_alone": cer_alone,
            "cer_yolo": cer_yolo,
            "cer_arabert": cer_arabert,
            "wer_alone": wer_alone,
            "wer_yolo": wer_yolo,
            "wer_arabert": wer_arabert,
        })

        print(f"[{i}/10] Image: {base_name[:12]}")
        print(f"  Ground Truth:    {gt_text}")
        print(f"  [1] EasyOCR Alone:  {res_alone.raw_text} (CER: {cer_alone:.4f})")
        print(f"  [2] YOLO + EasyOCR: {res_yolo.raw_text} (CER: {cer_yolo:.4f})")
        print(f"  [3] + AraBERT:     {arabert_text} (CER: {cer_arabert:.4f})")
        print("-" * 80)

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("  3-STAGE COMPARISON TABLE (10 Random New Images)")
    print("=" * 80)
    print(f"{'Idx':<4} {'Image ID':<10} {'Alone CER':>10} {'YOLO CER':>10} {'AraBERT CER':>12}")
    print("-" * 80)
    for r in records:
        print(f"{r['idx']:<4} {r['id']:<10} {r['cer_alone']:>10.4f} {r['cer_yolo']:>10.4f} {r['cer_arabert']:>12.4f}")

    avg_alone = sum(r["cer_alone"] for r in records) / len(records)
    avg_yolo = sum(r["cer_yolo"] for r in records) / len(records)
    avg_arabert = sum(r["cer_arabert"] for r in records) / len(records)

    print("================================================================================")
    print("  AVERAGE CER COMPARISON")
    print("================================================================================")
    print(f"  [1] EasyOCR Alone:             CER = {avg_alone:.4f}")
    print(f"  [2] YOLOv11s + EasyOCR:        CER = {avg_yolo:.4f}")
    print(f"  [3] YOLOv11s + EasyOCR + AraBERT: CER = {avg_arabert:.4f}  ★")
    print("================================================================================")

if __name__ == "__main__":
    main()
