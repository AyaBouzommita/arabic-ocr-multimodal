"""Run PaddleOCR on a Supervisely-format OCR archive dataset.

Processes images from an archive such as:
    archive (2)/Documents/Documents/{Category}/img/

Optionally evaluates CER/WER against "Transcription" tags in the
matching ann/*.json annotation files.

Usage:
    python -m scripts.run_paddleocr_archive --archive "C:/Downloads/archive (2)"
    python -m scripts.run_paddleocr_archive --archive "..." --category Invoice --limit 10
    python -m scripts.run_paddleocr_archive --archive "..." --evaluate --save-json
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.archive_loader import ArchiveDatasetLoader
from evaluation.metrics import compute_cer, compute_wer, corpus_summary
from ocr.paddle import PaddleOCREngine

import pandas as pd


def main():
    """Run PaddleOCR on the archive dataset."""
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on a Supervisely OCR archive dataset"
    )
    parser.add_argument(
        "--archive",
        required=True,
        help="Path to the archive root (e.g. Downloads/archive (2))",
    )
    parser.add_argument(
        "--output-dir",
        default="results/archive",
        help="Directory for CSV and JSON outputs (default: results/archive)",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Process only this category (repeatable). Default: all categories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute CER/WER for images that have Transcription annotations",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save per-document OCR JSON outputs",
    )
    parser.add_argument(
        "--lang",
        default="arabic",
        help="PaddleOCR language code (default: arabic)",
    )
    parser.add_argument(
        "--no-angle-cls",
        action="store_true",
        help="Disable text angle classification",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU inference",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Enable Otsu binarisation preprocessing",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising during preprocessing",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PADDLEOCR — Supervisely Archive Dataset")
    print("=" * 60)

    loader = ArchiveDatasetLoader(args.archive, categories=args.categories)
    summary = loader.summary()

    print(f"\n📂 Archive: {args.archive}")
    print(f"   Dataset root:  {loader.dataset_root}")
    print(f"   Categories:    {summary['categories']}")
    print(f"   Total images:  {summary['total_images']}")
    print(f"   With GT tags:  {summary['with_ground_truth']}")
    print(f"   Without GT:    {summary['without_ground_truth']}")

    pairs = loader.get_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"\n⚠️  Limiting run to first {args.limit} images")

    if not pairs:
        print("\n❌ No images found in the archive.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    if args.save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⏳ Loading PaddleOCR model (lang={args.lang})...")
    engine = PaddleOCREngine(
        lang=args.lang,
        use_angle_cls=not args.no_angle_cls,
        use_gpu=args.gpu,
        enable_denoise=not args.no_denoise,
        enable_threshold=args.threshold,
    )

    print(f"\n🚀 Running PaddleOCR on {len(pairs)} images...\n")

    records = []
    for i, (image_path, gt_text, category) in enumerate(pairs, start=1):
        doc_id = loader.document_id(image_path)
        result = engine.extract_text(str(image_path), document_id=doc_id)

        record = {
            "document_id": doc_id,
            "category": category,
            "image_path": str(image_path),
            "ground_truth": gt_text or "",
            "ocr_text": result.raw_text,
            "avg_confidence": result.avg_confidence,
            "processing_time_ms": result.processing_time_ms,
            "token_count": result.word_count,
        }

        if args.evaluate and gt_text:
            record["cer"] = compute_cer(gt_text, result.raw_text)
            record["wer"] = compute_wer(gt_text, result.raw_text)
        else:
            record["cer"] = None
            record["wer"] = None

        records.append(record)

        if args.save_json:
            safe_name = doc_id.replace("/", "__")
            json_path = json_dir / f"{safe_name}_paddleocr.json"
            json_path.write_text(result.to_json(), encoding="utf-8")

        gt_preview = (gt_text[:40] + "…") if gt_text and len(gt_text) > 40 else (gt_text or "—")
        ocr_preview = (
            (result.raw_text[:40] + "…")
            if len(result.raw_text) > 40
            else (result.raw_text or "—")
        )
        print(f"[{i}/{len(pairs)}] {doc_id}")
        print(f"         GT:  {gt_preview}")
        print(f"         OCR: {ocr_preview}")

    df = pd.DataFrame(records)
    csv_path = output_dir / "paddleocr_archive_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n📊 Results saved to: {csv_path}")

    if args.evaluate:
        eval_df = df[df["ground_truth"].astype(bool)].copy()
        eval_df = eval_df.dropna(subset=["cer"])
        if len(eval_df) > 0:
            agg = corpus_summary(eval_df)
            print("\n" + "=" * 60)
            print("  EVALUATION (images with Transcription tags)")
            print("=" * 60)
            print(f"  Documents evaluated:  {agg['total_documents']}")
            print(f"  Average CER:          {agg['avg_cer']:.4f}")
            print(f"  Average WER:          {agg['avg_wer']:.4f}")
            print(f"  Avg Confidence:       {agg['avg_confidence']:.1f}%")
            print("=" * 60)

            eval_csv = output_dir / "paddleocr_archive_eval.csv"
            eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
            print(f"📊 Evaluation CSV saved to: {eval_csv}")
        else:
            print("\n⚠️  --evaluate was set but no images had Transcription tags.")

    print("\n✅ Archive OCR run complete.")


if __name__ == "__main__":
    main()
