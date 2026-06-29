"""CER/WER evaluation metrics using JiWER.

Provides functions to compute Character Error Rate (CER) and Word Error
Rate (WER) between OCR output and ground-truth reference text. Also
includes a corpus-level evaluator that runs an OCR engine on all paired
documents and produces a results DataFrame.

Usage:
    from evaluation.metrics import compute_cer, compute_wer
    cer = compute_cer("الحياة جميلة", "الحياه جميله")
    wer = compute_wer("الحياة جميلة", "الحياه جميله")
"""

from typing import Dict, List

import jiwer
import pandas as pd

from evaluation.ground_truth import GroundTruthLoader
from ocr.base_engine import OCREngine


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis.

    CER = (S + D + I) / N, where S=substitutions, D=deletions,
    I=insertions at the character level, N=reference length.

    Args:
        reference: Ground-truth text.
        hypothesis: OCR-extracted text.

    Returns:
        CER as a float (0.0 = perfect, 1.0 = all wrong).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return round(jiwer.cer(reference, hypothesis), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N, where S=substitutions, D=deletions,
    I=insertions at the word level, N=reference word count.

    Args:
        reference: Ground-truth text.
        hypothesis: OCR-extracted text.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = all wrong).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return round(jiwer.wer(reference, hypothesis), 4)


def evaluate_document(gt_text: str, ocr_text: str) -> Dict:
    """Evaluate a single document — compute CER, WER, and summary stats.

    Args:
        gt_text: Ground-truth reference text.
        ocr_text: OCR-extracted hypothesis text.

    Returns:
        Dictionary with keys: cer, wer, ref_chars, hyp_chars,
        ref_words, hyp_words.
    """
    return {
        "cer": compute_cer(gt_text, ocr_text),
        "wer": compute_wer(gt_text, ocr_text),
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
        DataFrame with columns: document_id, image_path, ground_truth,
        ocr_text, cer, wer, avg_confidence, processing_time_ms, token_count.

    Raises:
        ValueError: If no image/ground-truth pairs are found.
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
                "avg_confidence": result.avg_confidence,
                "processing_time_ms": result.processing_time_ms,
                "token_count": result.word_count,
            }
        )

    return pd.DataFrame(records)


def corpus_summary(df: pd.DataFrame) -> Dict:
    """Compute aggregate metrics from a corpus evaluation DataFrame.

    Args:
        df: DataFrame produced by evaluate_corpus().

    Returns:
        Dictionary with avg_cer, avg_wer, total_documents, etc.
    """
    return {
        "avg_cer": round(df["cer"].mean(), 4),
        "avg_wer": round(df["wer"].mean(), 4),
        "median_cer": round(df["cer"].median(), 4),
        "median_wer": round(df["wer"].median(), 4),
        "min_cer": round(df["cer"].min(), 4),
        "max_cer": round(df["cer"].max(), 4),
        "total_documents": len(df),
        "total_time_ms": int(df["processing_time_ms"].sum()),
        "avg_confidence": round(df["avg_confidence"].mean(), 2),
    }
