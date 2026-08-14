"""YOLOv11s + PaddleOCR integrated pipeline vs PaddleOCR alone comparison.

This script:
1. Runs PaddleOCR alone on test images (baseline)
2. Runs YOLOv11s to detect text regions, crops them, then runs PaddleOCR on each crop
3. Compares CER/WER metrics side-by-side and saves results

Usage:
    python -m scripts.run_yolo_paddleocr_pipeline
    python -m scripts.run_yolo_paddleocr_pipeline --use-gpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from ocr.easyocr.engine import EasyOCREngine
from ocr.result import OCRResult, OCRToken
from ocr.postprocessing.bilingual_corrector import BilingualCorrector
from ocr.postprocessing.noise_filter import OCRNoiseFilter
from ocr.preprocessing import ImageEnhancer
from ocr.utils import sort_boxes_smart
from evaluation.ground_truth import GroundTruthLoader
from evaluation.metrics import compute_cer, compute_wer, corpus_summary


# ── Constants ────────────────────────────────────────────────────────────────

YOLO_WEIGHTS = "runs/detect/evaluation/yolo_comparison/yolov11s/weights/best.pt"

# Classes considered as "text" regions for OCR
TEXT_CLASSES = {"text", "table", "letters"}

# Classes for Arabic alphabet (individual character detections)
ARABIC_LETTER_CLASSES = {
    "aleph", "baaa", "daal", "dad", "faaa", "geem", "haaa", "hamza",
    "hamzasater", "kaaf", "lam", "mem", "non", "qaf", "raaa", "sad",
    "sen", "sheen", "taaa", "thaa", "thal", "then", "ttaa", "waaa",
    "yaaa", "zaaa", "3en", "5aaa", "5en",
}


def fix_arabic_direction(text: str) -> str:
    """Convert raw OCR character strings into 100% natural, readable connected Arabic text."""
    if not text or not text.strip():
        return text
    words = text.split()
    fixed_words = []
    for word in words:
        if any('\u0600' <= char <= '\u06FF' for char in word):
            fixed_words.append(word[::-1])
        else:
            fixed_words.append(word)
    # Reverse word order so reading direction flows right-to-left naturally
    return " ".join(fixed_words[::-1])


def run_easyocr_alone(engine, image_path, document_id=None):
    """Run EasyOCR directly on the full image (baseline)."""
    result = engine.extract_text(str(image_path), document_id=document_id)
    return result


def run_yolo_easyocr_pipeline(yolo_model, engine_ar_en, engine_fr_en, image_path, document_id=None, conf_thresh=0.25, enhancer=None, paddle_ar=None):
    """Run YOLOv11s → crop text regions → OCR on each crop.
    
    For Arabic documents: runs EasyOCR + PaddleOCR Arabic ensemble,
    picks the higher-confidence result per crop.
    For French/Latin documents: runs PaddleOCR v4 (French).
    
    Args:
        yolo_model: Loaded YOLO model.
        engine_ar_en: EasyOCR engine (ar+en) for Arabic text.
        engine_fr_en: PaddleOCR instance (lang='fr') for French/Latin text.
        image_path: Path to the input image.
        document_id: Optional document ID.
        conf_thresh: YOLO confidence threshold.
        enhancer: Optional ImageEnhancer instance.
        paddle_ar: Optional PaddleOCR instance (lang='ar') for ensemble.
    
    Returns:
        OCRResult with combined text from all detected text regions.
    """
    doc_id = document_id or Path(image_path).stem
    start_time = time.perf_counter()

    # Load the original image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = image.shape[:2]

    # Run YOLO detection
    results = yolo_model.predict(
        source=str(image_path),
        conf=conf_thresh,
        imgsz=640,
        verbose=False,
    )

    # Get class names from the model
    class_names = yolo_model.names  # {0: 'text', 1: 'table', ...}

    # Collect text region crops sorted top-to-bottom, left-to-right
    text_regions = []
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = class_names.get(cls_id, "unknown")
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # Mask out non-text regions (stamps, signatures, pictures, QR codes)
                # to prevent OCR from reading text inside them.
                if cls_name in {"stamp", "signature", "picture", "qr_code"}:
                    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                    x2_int, y2_int = min(w, int(x2)), min(h, int(y2))
                    # Fill with white (assuming document background is mostly white)
                    cv2.rectangle(image, (x1_int, y1_int), (x2_int, y2_int), (255, 255, 255), -1)
                    continue

                # Only process text-like regions and Arabic letters
                if cls_name in TEXT_CLASSES or cls_name in ARABIC_LETTER_CLASSES:
                    
                    # Clamp to image bounds and add small padding
                    pad = 5
                    x1 = max(0, int(x1) - pad)
                    y1 = max(0, int(y1) - pad)
                    x2 = min(w, int(x2) + pad)
                    y2 = min(h, int(y2) + pad)
                    
                    # Skip tiny regions (noise)
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    text_regions.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_name,
                        "conf": conf,
                        "crop": image[y1:y2, x1:x2].copy(),
                    })
                    
    # Filter out overlapping/nested boxes to prevent duplicate text extraction
    def filter_overlapping_boxes(regions, iou_thresh=0.4, iom_thresh=0.7):
        # Sort by confidence descending
        regions = sorted(regions, key=lambda x: x["conf"], reverse=True)
        keep = []
        for i, r1 in enumerate(regions):
            b1 = r1["bbox"]
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            is_redundant = False
            
            for r2 in keep:
                b2 = r2["bbox"]
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                
                # Intersection
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih
                
                if inter_area > 0:
                    iou = inter_area / (area1 + area2 - inter_area)
                    iom = inter_area / min(area1, area2)
                    
                    if iou > iou_thresh or iom > iom_thresh:
                        is_redundant = True
                        break
            
            if not is_redundant:
                keep.append(r1)
                
        return keep

    text_regions = filter_overlapping_boxes(text_regions)

    # We will process each crop with BOTH Arabic and French pipelines and pick the winner per-crop
    # Default to RTL sorting since it's an Arabic-focused application
    is_rtl = True
    
    # Sort regions Top-to-Bottom, then Right-to-Left
    text_regions = sort_boxes_smart(text_regions, lambda r: r["bbox"], is_rtl=is_rtl)

    all_tokens = []
    all_text_parts = []

        # Fallback: if YOLO found nothing, run full image through EasyOCR (Arabic/English)
        try:
            result = engine_ar_en.extract_text(str(image_path), document_id=doc_id)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            result.processing_time_ms = elapsed_ms
            result.engine = "easyocr_fallback"
            return result
        except Exception:
            pass

    for region in text_regions:
        crop = region["crop"]
        bbox_offset = region["bbox"]  # [x1, y1, x2, y2] in original image

        # Apply Arabic-optimized preprocessing (upscale, bilateral, CLAHE, unsharp)
        if enhancer is not None:
            enhanced_crop = enhancer.enhance_crop_arabic(crop)
        else:
            # Fallback: at minimum upscale small crops
            h_crop, w_crop = crop.shape[:2]
            if h_crop < 128:
                scale = 128 / h_crop
                enhanced_crop = cv2.resize(crop, (int(w_crop * scale), int(h_crop * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                enhanced_crop = crop
        
        # ── Engine A: EasyOCR (Arabic/English) ──
        easyocr_tokens = []
        easyocr_avg_conf = 0.0
        try:
            reader = engine_ar_en._get_reader()
            easyocr_results = reader.readtext(
                enhanced_crop,
                detail=1,
                paragraph=False,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.35,
                mag_ratio=1.5,
                decoder='beamsearch',
                beamWidth=10,
                slope_ths=0.2,
                add_margin=0.1,
            )
            
            def get_item_bbox(item):
                box_p = item[0]
                return [min(p[0] for p in box_p), min(p[1] for p in box_p),
                        max(p[0] for p in box_p), max(p[1] for p in box_p)]
            easyocr_results = sort_boxes_smart(easyocr_results, get_item_bbox, is_rtl=is_rtl)

            confs = []
            for result_item in easyocr_results:
                box_points = result_item[0]
                text = result_item[1]
                conf = result_item[2]
                from ocr.postprocessing.normalizer import normalize_arabic_numerals, normalize_arabic_text
                text = normalize_arabic_numerals(text)
                text = normalize_arabic_text(text)

                if not text.strip():
                    continue

                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                global_bbox = [
                    float(min(xs)) + bbox_offset[0],
                    float(min(ys)) + bbox_offset[1],
                    float(max(xs)) + bbox_offset[0],
                    float(max(ys)) + bbox_offset[1],
                ]

                easyocr_tokens.append(OCRToken(
                    text=text.strip(),
                    bbox=global_bbox,
                    confidence=float(conf) * 100.0,
                ))
                confs.append(float(conf))
            
            if confs:
                easyocr_avg_conf = sum(confs) / len(confs)
        except Exception:
            pass
        
        # ── Engine B: PaddleOCR (Arabic) ──
        paddle_ar_tokens = []
        paddle_ar_avg_conf = 0.0
        if paddle_ar is not None:
            try:
                paddle_results = paddle_ar.ocr(enhanced_crop, cls=True)
                if paddle_results and paddle_results[0]:
                    confs_p = []
                    for line in paddle_results[0]:
                        box_points = line[0]
                        text = line[1][0]
                        conf = line[1][1]
                        from ocr.postprocessing.normalizer import normalize_arabic_numerals, normalize_arabic_text
                        text = normalize_arabic_numerals(text)
                        text = normalize_arabic_text(text)

                        if not text.strip():
                            continue

                        xs = [p[0] for p in box_points]
                        ys = [p[1] for p in box_points]
                        global_bbox = [
                            float(min(xs)) + bbox_offset[0],
                            float(min(ys)) + bbox_offset[1],
                            float(max(xs)) + bbox_offset[0],
                            float(max(ys)) + bbox_offset[1],
                        ]

                        paddle_ar_tokens.append(OCRToken(
                            text=text.strip(),
                            bbox=global_bbox,
                            confidence=float(conf) * 100.0,
                        ))
                        confs_p.append(float(conf))
                    
                    if confs_p:
                        paddle_ar_avg_conf = sum(confs_p) / len(confs_p)
            except Exception:
                pass

        # ── Engine C: PaddleOCR (French/Latin) ──
        paddle_fr_tokens = []
        paddle_fr_avg_conf = 0.0
        try:
            paddle_results = engine_fr_en.ocr(crop, cls=True)
            if paddle_results and paddle_results[0]:
                confs_p = []
                for line in paddle_results[0]:
                    box_points = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    from ocr.postprocessing.normalizer import normalize_arabic_numerals
                    text = normalize_arabic_numerals(text)

                    if not text.strip():
                        continue

                    xs = [p[0] for p in box_points]
                    ys = [p[1] for p in box_points]
                    global_bbox = [
                        float(min(xs)) + bbox_offset[0],
                        float(min(ys)) + bbox_offset[1],
                        float(max(xs)) + bbox_offset[0],
                        float(max(ys)) + bbox_offset[1],
                    ]

                    paddle_fr_tokens.append(OCRToken(
                        text=text.strip(),
                        bbox=global_bbox,
                        confidence=float(conf) * 100.0,
                    ))
                    confs_p.append(float(conf))
                if confs_p:
                    paddle_fr_avg_conf = sum(confs_p) / len(confs_p)
        except Exception:
            pass
        
        # ── Per-Crop Language Detection & Routing ──
        # Determine if this crop actually contains Arabic text from the Arabic engines
        has_arabic_chars = False
        arabic_text_combined = " ".join([t.text for t in easyocr_tokens + paddle_ar_tokens])
        for char in arabic_text_combined:
            if '\u0600' <= char <= '\u06FF':
                has_arabic_chars = True
                break
                
        if has_arabic_chars:
            # It's an Arabic or mixed Arabic/English crop. Pick between EasyOCR and PaddleOCR_AR
            chosen_tokens = easyocr_tokens if easyocr_avg_conf >= paddle_ar_avg_conf else paddle_ar_tokens
        else:
            # No Arabic detected in this crop. It must be French/English. Use PaddleOCR_FR
            # (If French OCR also failed, fallback to whatever EasyOCR English picked up)
            if paddle_fr_tokens and paddle_fr_avg_conf > 20.0:
                chosen_tokens = paddle_fr_tokens
            else:
                chosen_tokens = easyocr_tokens
        
        for token in chosen_tokens:
            all_tokens.append(token)
            all_text_parts.append(token.text)

    # Filter out garbage text from decorative borders, stamps, and patterns
    noise_filter = OCRNoiseFilter()
    clean_tokens = noise_filter.filter_tokens(all_tokens)
    clean_text_parts = [t.text for t in clean_tokens]

    raw_text = " ".join(clean_text_parts)
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return OCRResult(
        document_id=doc_id,
        engine="yolo+easyocr+paddleocr",
        raw_text=raw_text,
        tokens=clean_tokens,
        processing_time_ms=elapsed_ms,
    )



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare PaddleOCR alone vs YOLOv11s + PaddleOCR pipeline"
    )
    parser.add_argument("--image-dir", default="data/raw", help="Image directory")
    parser.add_argument("--gt-dir", default="data/ground_truth", help="Ground truth directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--yolo-weights", default=YOLO_WEIGHTS, help="Path to YOLOv11s best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOv11s + PaddleOCR PIPELINE vs PaddleOCR ALONE")
    print("=" * 70)

    # ── Check CUDA ──
    print(f"\n  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")

    # ── Load models ──
    print("\n[Loading] YOLOv11s weights: {args.yolo_weights}")
    yolo_model = YOLO(args.yolo_weights)
    
    print("[Loading] EasyOCR engine (ar, en)...")
    easyocr_ar_en = EasyOCREngine(languages=["ar", "en"], gpu=args.use_gpu)
    print("[Loading] EasyOCR engine (fr, en)...")
    easyocr_fr_en = EasyOCREngine(languages=["fr", "en"], gpu=args.use_gpu)

    # ── Load ground truth ──
    loader = GroundTruthLoader(image_dir=args.image_dir, gt_dir=args.gt_dir)
    pairs = loader.get_pairs()
    summary = loader.summary()

    print(f"\n[Corpus] {args.image_dir}")
    print(f"   Total images:    {summary['total_images']}")
    print(f"   With ground GT:  {summary['paired']}")

    if summary["paired"] == 0:
        print("\n[ERROR] No image/ground-truth pairs found!")
        sys.exit(1)

    # ── Run both pipelines ──
    records_baseline = []
    records_pipeline = []

    print(f"\n[Running] Evaluating {summary['paired']} documents...\n")

    for image_path, gt_text in pairs:
        doc_id = image_path.stem
        print(f"  Processing: {doc_id}")

        # --- EasyOCR alone (baseline) ---
        print(f"    [1/2] EasyOCR alone...", end=" ")
        result_baseline = run_easyocr_alone(easyocr_ar_en, image_path, doc_id)
        cer_b = compute_cer(gt_text, result_baseline.raw_text)
        wer_b = compute_wer(gt_text, result_baseline.raw_text)
        print(f"CER={cer_b:.4f}  WER={wer_b:.4f}")

        records_baseline.append({
            "document_id": doc_id,
            "method": "EasyOCR Alone",
            "ground_truth": gt_text,
            "ocr_text": result_baseline.raw_text,
            "cer": cer_b,
            "wer": wer_b,
            "avg_confidence": result_baseline.avg_confidence,
            "processing_time_ms": result_baseline.processing_time_ms,
            "token_count": result_baseline.word_count,
        })

        # --- YOLO + EasyOCR pipeline ---
        print(f"    [2/2] YOLOv11s + EasyOCR...", end=" ")
        result_pipeline = run_yolo_easyocr_pipeline(
            yolo_model=yolo_model,
            engine_ar_en=easyocr_ar_en,
            engine_fr_en=easyocr_fr_en,
            image_path=image_path,
            document_id=doc_id,
            conf_thresh=args.conf,
            enhancer=None,
        )
        cer_p = compute_cer(gt_text, result_pipeline.raw_text)
        wer_p = compute_wer(gt_text, result_pipeline.raw_text)
        print(f"CER={cer_p:.4f}  WER={wer_p:.4f}")

        records_pipeline.append({
            "document_id": doc_id,
            "method": "YOLOv11s + EasyOCR",
            "ground_truth": gt_text,
            "ocr_text": result_pipeline.raw_text,
            "cer": cer_p,
            "wer": wer_p,
            "avg_confidence": result_pipeline.avg_confidence,
            "processing_time_ms": result_pipeline.processing_time_ms,
            "token_count": result_pipeline.word_count,
        })

        print()

    # ── Build DataFrames ──
    df_baseline = pd.DataFrame(records_baseline)
    df_pipeline = pd.DataFrame(records_pipeline)
    df_combined = pd.concat([df_baseline, df_pipeline], ignore_index=True)

    # ── Save results ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sprint2_yolo_pipeline_comparison.csv"
    df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")

    # ── Print comparison table ──
    print("\n" + "=" * 70)
    print("  COMPARISON: PaddleOCR Alone vs YOLOv11s + PaddleOCR")
    print("=" * 70)

    print(f"\n{'Document':<15} {'Method':<25} {'CER':>8} {'WER':>8} {'Conf':>8} {'Time(ms)':>10}")
    print("-" * 74)

    for _, row in df_combined.iterrows():
        print(
            f"{row['document_id']:<15} "
            f"{row['method']:<25} "
            f"{row['cer']:>8.4f} "
            f"{row['wer']:>8.4f} "
            f"{row['avg_confidence']:>8.1f} "
            f"{row['processing_time_ms']:>10}"
        )

    # ── Print aggregate summary ──
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    for method, df in [("PaddleOCR Alone", df_baseline), ("YOLOv11s + PaddleOCR", df_pipeline)]:
        avg_cer = df["cer"].mean()
        avg_wer = df["wer"].mean()
        avg_conf = df["avg_confidence"].mean()
        total_time = df["processing_time_ms"].sum()

        print(f"\n  [{method}]")
        print(f"    Average CER:     {avg_cer:.4f} ({avg_cer*100:.1f}%)")
        print(f"    Average WER:     {avg_wer:.4f} ({avg_wer*100:.1f}%)")
        print(f"    Avg Confidence:  {avg_conf:.1f}%")
        print(f"    Total Time:      {total_time} ms")

    # ── Improvement calculation ──
    baseline_cer = df_baseline["cer"].mean()
    pipeline_cer = df_pipeline["cer"].mean()
    baseline_wer = df_baseline["wer"].mean()
    pipeline_wer = df_pipeline["wer"].mean()

    if baseline_cer > 0:
        cer_improvement = ((baseline_cer - pipeline_cer) / baseline_cer) * 100
    else:
        cer_improvement = 0.0

    if baseline_wer > 0:
        wer_improvement = ((baseline_wer - pipeline_wer) / baseline_wer) * 100
    else:
        wer_improvement = 0.0

    print("\n" + "=" * 70)
    print("  IMPROVEMENT WITH YOLO")
    print("=" * 70)
    print(f"  CER: {baseline_cer:.4f} -> {pipeline_cer:.4f}  ({cer_improvement:+.1f}% {'improvement' if cer_improvement > 0 else 'regression'})")
    print(f"  WER: {baseline_wer:.4f} -> {pipeline_wer:.4f}  ({wer_improvement:+.1f}% {'improvement' if wer_improvement > 0 else 'regression'})")
    print("=" * 70)

    # ── Print OCR output samples ──
    print("\n" + "=" * 70)
    print("  OCR OUTPUT SAMPLES")
    print("=" * 70)

    for _, row in df_combined.iterrows():
        print(f"\n  [{row['method']}] {row['document_id']}:")
        print(f"    Ground Truth: {row['ground_truth']}")
        print(f"    OCR Output:   {row['ocr_text']}")
        print(f"    CER: {row['cer']:.4f}  |  WER: {row['wer']:.4f}")

    print(f"\n[DONE] Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
