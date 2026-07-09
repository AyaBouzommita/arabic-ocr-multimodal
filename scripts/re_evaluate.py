import sys
import re
from pathlib import Path
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics import (
    evaluate_document,
    corpus_summary
)


def fix_arabic_bidi(text: str) -> str:
    """Fix bidirectional Arabic text by reversing Arabic word sequences character-by-character."""
    if not text:
        return ""
    # Match sequences of Arabic characters and spaces between them
    arabic_char = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
    arabic_block = f"{arabic_char}+(?:\s+{arabic_char}+)*"
    return re.sub(arabic_block, lambda m: m.group(0)[::-1], text)


def main():
    results_csv = project_root / "results" / "archive" / "paddleocr_archive_results.csv"
    if not results_csv.exists():
        print(f"Error: Results file not found at {results_csv}")
        sys.exit(1)

    print(f"Reading results from {results_csv}...")
    df = pd.read_csv(results_csv)

    # Filter rows with ground truth
    df_eval = df[df["ground_truth"].notna() & (df["ground_truth"].str.strip() != "")].copy()

    print(f"Found {len(df_eval)} annotated documents to re-evaluate.")

    cers, wers = [], []
    cers_norm, wers_norm = [], []
    cers_align, wers_align = [], []
    fixed_ocr_texts = []

    total = len(df_eval)
    for idx, (i, row) in enumerate(df_eval.iterrows(), 1):
        gt = str(row["ground_truth"])
        ocr = str(row["ocr_text"]) if pd.notna(row["ocr_text"]) else ""
        
        ocr_fixed = fix_arabic_bidi(ocr)
        fixed_ocr_texts.append(ocr_fixed)

        metrics = evaluate_document(gt, ocr_fixed)

        cers.append(metrics["cer"])
        wers.append(metrics["wer"])
        cers_norm.append(metrics["cer_normalized"])
        wers_norm.append(metrics["wer_normalized"])
        cers_align.append(metrics["cer_aligned"])
        wers_align.append(metrics["wer_aligned"])

        if idx % 500 == 0 or idx == total:
            print(f"Processed {idx}/{total} documents...")

    df_eval["ocr_text_fixed"] = fixed_ocr_texts
    df_eval["cer"] = cers
    df_eval["wer"] = wers

    df_eval["cer_normalized"] = cers_norm
    df_eval["wer_normalized"] = wers_norm
    df_eval["cer_aligned"] = cers_align
    df_eval["wer_aligned"] = wers_align

    # Save the updated evaluation CSV
    output_csv = project_root / "results" / "archive" / "paddleocr_archive_eval_v2.csv"
    df_eval.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n📊 New evaluation results saved to: {output_csv}")

    # Generate summary
    summary = corpus_summary(df_eval)

    print("\n" + "=" * 60)
    print("  RE-EVALUATION SUMMARY (Improved Metrics)")
    print("=" * 60)
    print(f"  Documents evaluated:  {summary['total_documents']}")
    print("-" * 60)
    print("  RAW METRICS:")
    print(f"    Average CER:          {summary['avg_cer']:.4f} ({summary['avg_cer']*100:.2f}%)")
    print(f"    Average WER:          {summary['avg_wer']:.4f} ({summary['avg_wer']*100:.2f}%)")
    print("-" * 60)
    print("  NORMALIZED METRICS:")
    print(f"    Average CER:          {summary['avg_cer_normalized']:.4f} ({summary['avg_cer_normalized']*100:.2f}%)")
    print(f"    Average WER:          {summary['avg_wer_normalized']:.4f} ({summary['avg_wer_normalized']*100:.2f}%)")
    print("-" * 60)
    print("  SEGMENT-ALIGNED METRICS (Recall-focused for Partial GT):")
    print(f"    Average CER:          {summary['avg_cer_aligned']:.4f} ({summary['avg_cer_aligned']*100:.2f}%)")
    print(f"    Average WER:          {summary['avg_wer_aligned']:.4f} ({summary['avg_wer_aligned']*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
