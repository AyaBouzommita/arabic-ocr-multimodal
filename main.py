"""Arabic OCR Multimodal Pipeline — Main Entry Point.

Quick CLI to run Tesseract on a single image and print the structured
JSON output conforming to the interface contract.

Usage:
    python main.py data/raw/sample.png
    python main.py data/raw/sample2.png --lang ara+eng
    python main.py data/raw/sample.png --evaluate
"""

import argparse
import sys
from pathlib import Path

from ocr.tesseract import TesseractEngine
from evaluation.metrics import compute_cer, compute_wer
from evaluation.ground_truth import GroundTruthLoader


def main():
    """Run Tesseract OCR on a single image and print structured output."""
    parser = argparse.ArgumentParser(
        description="Arabic OCR — extract text from a document image"
    )
    parser.add_argument(
        "image",
        help="Path to the input image file",
    )
    parser.add_argument(
        "--lang",
        default="ara",
        help="Tesseract language code (default: ara)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help="Tesseract Page Segmentation Mode (default: 3)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also compute CER/WER against ground truth if available",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Print only the raw text (no JSON structure)",
    )
    args = parser.parse_args()

    engine = TesseractEngine(lang=args.lang, psm=args.psm)

    if args.simple:
        text = engine.extract_text_simple(args.image)
        print(text)
        return

    result = engine.extract_text(args.image)

    # Print structured JSON output
    print(result.to_json())

    # Optionally evaluate against ground truth
    if args.evaluate:
        gt_dir = Path("data/ground_truth")
        stem = Path(args.image).stem
        gt_path = gt_dir / f"{stem}.txt"

        if gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            cer = compute_cer(gt_text, result.raw_text)
            wer = compute_wer(gt_text, result.raw_text)
            print(f"\n--- Evaluation ---")
            print(f"Ground Truth: {gt_text}")
            print(f"OCR Output:   {result.raw_text}")
            print(f"CER: {cer:.4f}  ({cer*100:.1f}%)")
            print(f"WER: {wer:.4f}  ({wer*100:.1f}%)")
        else:
            print(f"\n⚠️  No ground truth found at {gt_path}")


if __name__ == "__main__":
    main()
