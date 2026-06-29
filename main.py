"""Arabic OCR Multimodal Pipeline — Main Entry Point (PaddleOCR).

Quick CLI to run PaddleOCR on a single image and print the structured
JSON output conforming to the interface contract.

Usage:
    python main.py data/raw/sample.png
    python main.py data/raw/sample.png --lang arabic
    python main.py data/raw/sample.png --evaluate
    python main.py data/raw/sample.png --simple
"""

import argparse
import sys
from pathlib import Path

from ocr.paddle import PaddleOCREngine
from evaluation.metrics import compute_cer, compute_wer


def main():
    """Run PaddleOCR on a single image and print structured output."""
    parser = argparse.ArgumentParser(
        description="Arabic OCR (PaddleOCR) — extract text from a document image"
    )
    parser.add_argument(
        "image",
        help="Path to the input image file",
    )
    parser.add_argument(
        "--lang",
        default="arabic",
        help="PaddleOCR language code (default: arabic)",
    )
    parser.add_argument(
        "--no-angle-cls",
        action="store_true",
        help="Disable angle classification",
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

    print("⏳ Loading PaddleOCR model...")
    engine = PaddleOCREngine(
        lang=args.lang,
        use_angle_cls=not args.no_angle_cls,
    )

    if args.simple:
        text = engine.extract_text_simple(args.image)
        print(text)
        return

    result = engine.extract_text(args.image)
    print(result.to_json())

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
