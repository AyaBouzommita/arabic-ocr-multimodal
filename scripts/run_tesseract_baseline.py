"""Run Tesseract Arabic baseline evaluation on the full corpus.

This script is the primary deliverable for US-04:
    'As the assigned engineer, I implement Tesseract Arabic baseline
     and measure CER/WER'

It runs Tesseract on every image in data/raw/ that has a matching
ground-truth file in data/ground_truth/, computes CER/WER metrics,
and saves:
    - results/sprint1_tesseract_baseline.csv  (per-document results)
    - results/sprint1_baseline.csv            (same, DoD reference name)
    - Console summary with aggregate metrics

Usage:
    python -m scripts.run_tesseract_baseline
    python -m scripts.run_tesseract_baseline --image-dir data/raw --gt-dir data/ground_truth
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr.tesseract import TesseractEngine
from evaluation.ground_truth import GroundTruthLoader
from evaluation.metrics import evaluate_corpus, corpus_summary


def main():
    """Run the Tesseract baseline evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Tesseract Arabic OCR baseline and measure CER/WER"
    )
    parser.add_argument(
        "--image-dir",
        default="data/raw",
        help="Directory containing document images (default: data/raw)",
    )
    parser.add_argument(
        "--gt-dir",
        default="data/ground_truth",
        help="Directory containing ground-truth .txt files (default: data/ground_truth)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results CSV (default: results)",
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
        "--threshold",
        action="store_true",
        help="Enable Otsu binary thresholding during preprocessing",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising during preprocessing",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save per-document OCR JSON outputs to results/json/",
    )
    args = parser.parse_args()

    # --- Setup ---
    print("=" * 60)
    print("  TESSERACT ARABIC BASELINE — Sprint 1 (US-04)")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = GroundTruthLoader(
        image_dir=args.image_dir,
        gt_dir=args.gt_dir,
    )

    # Print corpus summary
    summary = loader.summary()
    print(f"\n📂 Corpus: {args.image_dir}")
    print(f"   Total images:    {summary['total_images']}")
    print(f"   With ground GT:  {summary['paired']}")
    print(f"   Without GT:      {summary['unpaired']}")

    if summary["paired"] == 0:
        print("\n❌ No image/ground-truth pairs found. Cannot evaluate.")
        print("   Make sure each data/raw/foo.png has a data/ground_truth/foo.txt")
        sys.exit(1)

    unpaired = loader.get_unpaired_images()
    if unpaired:
        print(f"\n⚠️  Skipping {len(unpaired)} images without ground truth:")
        for img in unpaired:
            print(f"     - {img.name}")

    # --- Initialize Tesseract engine ---
    engine = TesseractEngine(
        lang=args.lang,
        psm=args.psm,
        enable_denoise=not args.no_denoise,
        enable_threshold=args.threshold,
    )
    print(f"\n🔧 Engine: Tesseract (lang={args.lang}, psm={args.psm})")
    print(f"   Preprocessing: grayscale=True, denoise={not args.no_denoise}, "
          f"threshold={args.threshold}")

    # --- Run evaluation ---
    print(f"\n🚀 Running Tesseract on {summary['paired']} documents...\n")

    df = evaluate_corpus(engine, loader)

    # --- Save per-document results ---
    csv_path = output_dir / "sprint1_tesseract_baseline.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📊 Per-document results saved to: {csv_path}")

    # Also save as sprint1_baseline.csv (DoD reference name)
    baseline_path = output_dir / "sprint1_baseline.csv"
    df.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    print(f"📊 Baseline CSV saved to: {baseline_path}")

    # --- Save JSON outputs if requested ---
    if args.save_json:
        json_dir = output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            engine_result = engine.extract_text(
                row["image_path"], document_id=row["document_id"]
            )
            json_path = json_dir / f"{row['document_id']}_tesseract.json"
            json_path.write_text(engine_result.to_json(), encoding="utf-8")
        print(f"📄 JSON outputs saved to: {json_dir}")

    # --- Print per-document results ---
    print("\n" + "─" * 60)
    print(f"{'Document':<20} {'CER':>8} {'WER':>8} {'Conf':>8} {'Time(ms)':>10}")
    print("─" * 60)

    for _, row in df.iterrows():
        print(
            f"{row['document_id']:<20} "
            f"{row['cer']:>8.4f} "
            f"{row['wer']:>8.4f} "
            f"{row['avg_confidence']:>8.1f} "
            f"{row['processing_time_ms']:>10}"
        )

    # --- Print aggregate summary ---
    agg = corpus_summary(df)
    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Documents evaluated:  {agg['total_documents']}")
    print(f"  Average CER:          {agg['avg_cer']:.4f} ({agg['avg_cer']*100:.1f}%)")
    print(f"  Average WER:          {agg['avg_wer']:.4f} ({agg['avg_wer']*100:.1f}%)")
    print(f"  Median CER:           {agg['median_cer']:.4f}")
    print(f"  Median WER:           {agg['median_wer']:.4f}")
    print(f"  Best CER:             {agg['min_cer']:.4f}")
    print(f"  Worst CER:            {agg['max_cer']:.4f}")
    print(f"  Avg Confidence:       {agg['avg_confidence']:.1f}%")
    print(f"  Total Time:           {agg['total_time_ms']} ms")
    print("=" * 60)

    # --- Print per-document OCR output for inspection ---
    print("\n" + "=" * 60)
    print("  OCR OUTPUT SAMPLES (for error analysis)")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"\n📄 {row['document_id']}:")
        print(f"   Ground Truth:  {row['ground_truth']}")
        print(f"   OCR Output:    {row['ocr_text']}")
        print(f"   CER: {row['cer']:.4f}  |  WER: {row['wer']:.4f}")

    print("\n✅ Tesseract baseline evaluation complete.")
    print(f"   Results in: {csv_path}")
    return df


if __name__ == "__main__":
    main()
