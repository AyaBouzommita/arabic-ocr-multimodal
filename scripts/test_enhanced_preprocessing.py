"""Test Enhanced Preprocessing (2.5x Upscaling + Sharpening) + Smart AraBERT Correction."""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr.paddleocr.engine import PaddleOCREngine
from ocr.postprocessing.arabert_corrector import AraBERTCorrector, levenshtein_distance
from evaluation.metrics import compute_cer, compute_wer

def preprocess_crop(crop_img):
    """Enhance image for OCR: 2.5x upscale + contrast enhancement."""
    if crop_img is None or crop_img.size == 0:
        return crop_img

    # 1. Upscale image 2.5x using bicubic interpolation
    h, w = crop_img.shape[:2]
    upscaled = cv2.resize(crop_img, (int(w * 2.5), int(h * 2.5)), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale & apply CLAHE (Adaptive Histogram Equalization)
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Convert back to 3-channel for PaddleOCR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def smart_arabert_correct(corrector, ocr_results_with_conf, max_edit_dist=1):
    """Smart AraBERT correction: ONLY touch low-confidence or candidate typo words."""
    words_and_confs = []
    for line in ocr_results_with_conf:
        word = line['text']
        conf = line['conf']
        words_and_confs.append((word, conf))

    full_words = [w[0] for w in words_and_confs]
    corrected_words = full_words.copy()

    for idx, (word, conf) in enumerate(words_and_confs):
        # ONLY touch words with low confidence (< 85%) or containing obvious non-Arabic noise
        if conf < 85.0 or not corrector.is_arabic_word(word):
            new_word, changed = corrector.correct_word_in_context(
                full_words, idx, top_k=10, max_edit_dist=max_edit_dist
            )
            if changed:
                corrected_words[idx] = new_word

    return " ".join(corrected_words)


def main():
    print("=" * 75)
    print("  TESTING ENHANCED PREPROCESSING (2.5x Upscaling + CLAHE) + SMART ARABERT")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paddle_engine = PaddleOCREngine(lang="ar", use_gpu=torch.cuda.is_available())
    corrector = AraBERTCorrector(device=device)

    # Test on Sample 3 (01155882)
    img_path = Path("data/OCR_GS_Data/ara/abhath/1524814015-ed4fea9f-2806-45f4-baad-cef0e4761e1b-png-kraken/01155882-c023-43d9-be49-ffe1b7ea5a2d.png")
    gt_path = Path("data/OCR_GS_Data/ara/abhath/1524814015-ed4fea9f-2806-45f4-baad-cef0e4761e1b-png-kraken/01155882-c023-43d9-be49-ffe1b7ea5a2d.gt.txt")
    
    gt_text = gt_path.read_text(encoding="utf-8").strip()
    img = cv2.imread(str(img_path))

    # Raw PaddleOCR
    res_raw = paddle_engine.extract_text(str(img_path))
    cer_raw = compute_cer(gt_text, res_raw.raw_text)

    # Enhanced Preprocessed PaddleOCR
    enhanced_img = preprocess_crop(img)
    temp_path = "data/temp_enhanced.png"
    cv2.imwrite(temp_path, enhanced_img)
    res_enhanced = paddle_engine.extract_text(temp_path)
    cer_enhanced = compute_cer(gt_text, res_enhanced.raw_text)

    print(f"\nGround Truth:  {gt_text}")
    print(f"Raw OCR:       {res_raw.raw_text} (CER: {cer_raw:.4f})")
    print(f"Enhanced OCR:  {res_enhanced.raw_text} (CER: {cer_enhanced:.4f})")
    print("=" * 75)

if __name__ == "__main__":
    main()
