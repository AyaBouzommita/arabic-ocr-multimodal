"""CER/WER evaluation metrics using JiWER with support for Arabic normalization and segment alignment.

Provides functions to compute Character Error Rate (CER) and Word Error
Rate (WER) between OCR output and ground-truth reference text. Also
includes a corpus-level evaluator that runs an OCR engine on all paired
documents and produces a results DataFrame.
"""

import re
from typing import Dict, List

import jiwer
import pandas as pd

from evaluation.ground_truth import GroundTruthLoader
from ocr.base_engine import OCREngine


def normalize_text(text: str) -> str:
    """Normalize Arabic text for fairer evaluation.

    - Converts to lowercase (for multilingual parts)
    - Removes Arabic diacritics/tashkeel (harakat) — vowel marks that are
      often present in GT but absent in OCR output (or vice versa)
    - Normalizes Arabic letters:
        - [أإآٱ] -> ا (Alef)
        - ة -> ه (Teh Marbuta -> Heh)
        - ى -> ي (Yeh/Alef Maksura -> Yeh)
    - Converts Hindi/Arabic digits (٠١٢٣٤٥٦٧٨٩) to standard digits (0123456789)
    - Removes punctuation and symbols
    - Normalizes multiple spaces to a single space
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove Arabic diacritics/Tashkeel (harakat):
    # U+0610-U+061A: Arabic sign sanah, etc.
    # U+064B-U+065F: Fathatan, Dammatan, Kasratan, Fathah, Dammah, Kasrah,
    #                Shadda, Sukun, and other combining marks
    # U+0670      : Arabic Letter Superscript Alef
    # U+06D6-U+06DC: Arabic Small High marks
    # U+06DF-U+06E4: Arabic Small High marks
    # U+06E7-U+06E8: Arabic Small High marks
    # U+06EA-U+06ED: Arabic Phrase marks
    tashkeel_pattern = (
        r"[\u0610-\u061A"
        r"\u064B-\u065F"
        r"\u0670"
        r"\u06D6-\u06DC"
        r"\u06DF-\u06E4"
        r"\u06E7-\u06E8"
        r"\u06EA-\u06ED]"
    )
    text = re.sub(tashkeel_pattern, "", text)

    # Normalization of letters
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)

    # Convert Hindi-Arabic digits to Western Arabic digits
    hindi_to_western = {
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"
    }
    for hindi, western in hindi_to_western.items():
        text = text.replace(hindi, western)

    # Remove punctuation, symbols, and extra characters (retaining words and digits)
    # This keeps Arabic letters, English letters, and numbers
    text = re.sub(r"[^\w\s\u0621-\u064A]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_best_aligning_segment(reference: str, hypothesis: str) -> str:
    """Find a contiguous word-level segment in the hypothesis that best matches the reference.

    This helps mitigate the issue of partial ground truth annotations by finding
    the substring of the full OCR output that is most similar to the GT.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words or not hyp_words:
        return hypothesis

    n_ref = len(ref_words)
    n_hyp = len(hyp_words)

    # If the hypothesis is shorter or same length, return the full normalized hypothesis
    if n_hyp <= n_ref:
        return hypothesis

    best_segment = hypothesis
    min_distance = float("inf")

    # Scan windows in the hypothesis. We allow the window size to vary slightly
    # (from n_ref to n_ref + 4 words) to handle insertions/deletions.
    for size in range(n_ref, min(n_ref + 5, n_hyp + 1)):
        for i in range(n_hyp - size + 1):
            segment_words = hyp_words[i : i + size]
            segment_str = " ".join(segment_words)
            try:
                # Calculate word-level distance
                dist = jiwer.wer(ref_norm, segment_str)
                if dist < min_distance:
                    min_distance = dist
                    # Map back to the original (non-normalized) words if possible,
                    # but since word alignment is tricky, returning the normalized segment
                    # is clean and sufficient for normalized evaluation.
                    best_segment = segment_str
            except Exception:
                pass

    return best_segment


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis.

    CER = (S + D + I) / N, where S=substitutions, D=deletions,
    I=insertions at the character level, N=reference length.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return round(jiwer.cer(reference, hypothesis), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N, where S=substitutions, D=deletions,
    I=insertions at the word level, N=reference word count.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return round(jiwer.wer(reference, hypothesis), 4)


def compute_cer_normalized(reference: str, hypothesis: str) -> float:
    """Compute CER after applying Arabic text normalization."""
    return compute_cer(normalize_text(reference), normalize_text(hypothesis))


def compute_wer_normalized(reference: str, hypothesis: str) -> float:
    """Compute WER after applying Arabic text normalization."""
    return compute_wer(normalize_text(reference), normalize_text(hypothesis))


def compute_cer_aligned(reference: str, hypothesis: str) -> float:
    """Compute CER on the best-aligning segment of the hypothesis (recall-focused)."""
    best_segment = find_best_aligning_segment(reference, hypothesis)
    return compute_cer(normalize_text(reference), best_segment)


def compute_wer_aligned(reference: str, hypothesis: str) -> float:
    """Compute WER on the best-aligning segment of the hypothesis (recall-focused)."""
    best_segment = find_best_aligning_segment(reference, hypothesis)
    return compute_wer(normalize_text(reference), best_segment)


def evaluate_document(gt_text: str, ocr_text: str) -> Dict:
    """Evaluate a single document — compute raw, normalized, and aligned CER/WER.

    Args:
        gt_text: Ground-truth reference text.
        ocr_text: OCR-extracted hypothesis text.

    Returns:
        Dictionary with raw, normalized, and aligned metrics.
    """
    return {
        "cer": compute_cer(gt_text, ocr_text),
        "wer": compute_wer(gt_text, ocr_text),
        "cer_normalized": compute_cer_normalized(gt_text, ocr_text),
        "wer_normalized": compute_wer_normalized(gt_text, ocr_text),
        "cer_aligned": compute_cer_aligned(gt_text, ocr_text),
        "wer_aligned": compute_wer_aligned(gt_text, ocr_text),
        "ref_chars": len(gt_text),
        "hyp_chars": len(ocr_text),
        "ref_words": len(gt_text.split()),
        "hyp_words": len(ocr_text.split()),
    }


def evaluate_corpus(
    engine: OCREngine,
    loader: GroundTruthLoader,
) -> pd.DataFrame:
    """Run a full corpus evaluation: OCR each paired image, compute CER/WER.

    Args:
        engine: The OCR engine to evaluate (must implement OCREngine).
        loader: Ground truth loader with paired images.

    Returns:
        DataFrame with columns containing both raw and improved evaluation metrics.
    """
    pairs = loader.get_pairs()
    if not pairs:
        raise ValueError(
            "No image/ground-truth pairs found. "
            "Ensure data/raw/ images have matching data/ground_truth/*.txt files."
        )

    records = []
    for image_path, gt_text in pairs:
        doc_id = image_path.stem
        result = engine.extract_text(str(image_path), document_id=doc_id)
        metrics = evaluate_document(gt_text, result.raw_text)

        records.append(
            {
                "document_id": doc_id,
                "image_path": str(image_path),
                "ground_truth": gt_text,
                "ocr_text": result.raw_text,
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "cer_normalized": metrics["cer_normalized"],
                "wer_normalized": metrics["wer_normalized"],
                "cer_aligned": metrics["cer_aligned"],
                "wer_aligned": metrics["wer_aligned"],
                "avg_confidence": result.avg_confidence,
                "processing_time_ms": result.processing_time_ms,
                "token_count": result.word_count,
            }
        )

    return pd.DataFrame(records)


def corpus_summary(df: pd.DataFrame) -> Dict:
    """Compute aggregate metrics from a corpus evaluation DataFrame.

    Args:
        df: DataFrame produced by evaluate_corpus() or containing evaluation metrics.

    Returns:
        Dictionary with avg_cer, avg_wer, normalized and aligned aggregates.
    """
    summary = {
        "total_documents": len(df),
        "total_time_ms": int(df["processing_time_ms"].sum()) if "processing_time_ms" in df.columns else 0,
        "avg_confidence": round(df["avg_confidence"].mean(), 2) if "avg_confidence" in df.columns else 0.0,
    }

    # Raw metrics
    if "cer" in df.columns:
        summary["avg_cer"] = round(df["cer"].mean(), 4)
        summary["median_cer"] = round(df["cer"].median(), 4)
    if "wer" in df.columns:
        summary["avg_wer"] = round(df["wer"].mean(), 4)
        summary["median_wer"] = round(df["wer"].median(), 4)

    # Normalized metrics
    if "cer_normalized" in df.columns:
        summary["avg_cer_normalized"] = round(df["cer_normalized"].mean(), 4)
        summary["median_cer_normalized"] = round(df["cer_normalized"].median(), 4)
    if "wer_normalized" in df.columns:
        summary["avg_wer_normalized"] = round(df["wer_normalized"].mean(), 4)
        summary["median_wer_normalized"] = round(df["wer_normalized"].median(), 4)

    # Segment Aligned (Recall) metrics
    if "cer_aligned" in df.columns:
        summary["avg_cer_aligned"] = round(df["cer_aligned"].mean(), 4)
        summary["median_cer_aligned"] = round(df["cer_aligned"].median(), 4)
    if "wer_aligned" in df.columns:
        summary["avg_wer_aligned"] = round(df["wer_aligned"].mean(), 4)
        summary["median_wer_aligned"] = round(df["wer_aligned"].median(), 4)

    return summary
