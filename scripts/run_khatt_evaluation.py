"""Run evaluation for Tesseract, EasyOCR, and PaddleOCR on the KHATT dataset."""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr.tesseract.engine import TesseractEngine
from ocr.easyocr.engine import EasyOCREngine
from ocr.paddleocr.engine import PaddleOCREngine
from evaluation.metrics import compute_cer, compute_wer
from scripts.khatt_decoder import load_khatt_ground_truth

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR engines on KHATT dataset")
    parser.add_argument("--image-dir", default="data/ar-data-groundtruth/Validate_deskewed/Validate_deskewed")
    parser.add_argument("--gt-csv", default="data/ar-data-groundtruth/Validation.csv")
    parser.add_argument("--limit", type=int, default=50, help="Max images to evaluate")
    args = parser.parse_args()

    # Load ground truth
    print("Loading KHATT ground truth...")
    gt_dict = load_khatt_ground_truth(args.gt_csv)
    print(f"Loaded {len(gt_dict)} ground truth labels.")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    images = sorted(images)[:args.limit]
    print(f"Found {len(images)} images to evaluate (limited to {args.limit}).")

    # Initialize engines
    print("\nInitializing OCR engines...")
    engines = {
        "Tesseract": TesseractEngine(lang="ara", psm=3),
        "EasyOCR": EasyOCREngine(languages=["ar"]),
        "PaddleOCR": PaddleOCREngine(lang="ar")
    }

    results = []

    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        gt_text = gt_dict.get(stem)
        
        if not gt_text:
            continue

        # Print simple progress line instead of full Arabic text to keep terminal clean
        print(f"Processing [{i}/{len(images)}] {img_path.name}...")

        for name, engine in engines.items():
            try:
                result = engine.extract_text(str(img_path))
                ocr_text = result.raw_text.strip()
                
                cer = compute_cer(gt_text, ocr_text)
                wer = compute_wer(gt_text, ocr_text)
                
                results.append({
                    "document_id": stem,
                    "engine": name,
                    "ground_truth": gt_text,
                    "ocr_text": ocr_text,
                    "cer": cer,
                    "wer": wer,
                    "time_ms": result.processing_time_ms
                })
            except Exception as e:
                print(f"  {name:10} | Error: {str(e)}")

    if not results:
        print("No results collected.")
        return

    # Analyze results
    df = pd.DataFrame(results)
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "khatt_evaluation_results.csv", index=False, encoding="utf-8-sig")
    
    print("\n" + "="*60)
    print("  FINAL EVALUATION SCORES (100 Images)")
    print("="*60)
    summary = df.groupby("engine").agg({
        "cer": ["mean"],
        "wer": ["mean"],
        "time_ms": ["mean"]
    }).reset_index()
    
    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    # Print formatted table (Convert to percentages for readability)
    print(f"{'Engine':<12} | {'Avg CER':<10} | {'Avg WER':<10} | {'Avg Time(ms)':<12}")
    print("-" * 52)
    for _, row in summary.iterrows():
        cer_pct = row['cer_mean'] * 100
        wer_pct = row['wer_mean'] * 100
        print(f"{row['engine']:<12} | {cer_pct:>8.2f}% | {wer_pct:>8.2f}% | {row['time_ms_mean']:>10.1f} ms")

    print("="*60)
    print("Evaluation complete. Detailed results saved to results/khatt_evaluation_results.csv")

if __name__ == "__main__":
    main()
