"""Tests for the evaluation pipeline — ground truth loader and CER/WER scorer."""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.ground_truth import GroundTruthLoader
from evaluation.metrics import compute_cer, compute_wer, evaluate_document


# ---------------------------------------------------------------------------
# CER / WER tests
# ---------------------------------------------------------------------------


class TestCERWER:
    """Tests for CER and WER computation."""

    def test_perfect_match_cer(self):
        """Identical strings should give CER = 0."""
        assert compute_cer("الحياة جميلة", "الحياة جميلة") == 0.0

    def test_perfect_match_wer(self):
        """Identical strings should give WER = 0."""
        assert compute_wer("الحياة جميلة", "الحياة جميلة") == 0.0

    def test_completely_wrong_cer(self):
        """Completely different text should give high CER."""
        cer = compute_cer("أ", "ب")
        assert cer > 0.0

    def test_substitution_cer(self):
        """Single character substitution should produce non-zero CER."""
        # 'ة' replaced with 'ه' — a common Arabic OCR error
        cer = compute_cer("الحياة", "الحياه")
        assert cer > 0.0
        assert cer < 1.0

    def test_insertion_wer(self):
        """Extra word should produce non-zero WER."""
        wer = compute_wer("الحياة جميلة", "الحياة جميلة جدا")
        assert wer > 0.0

    def test_empty_reference_with_empty_hypothesis(self):
        """Both empty strings should give 0 error."""
        assert compute_cer("", "") == 0.0
        assert compute_wer("", "") == 0.0

    def test_empty_reference_with_non_empty_hypothesis(self):
        """Empty reference with non-empty hypothesis should give 1.0."""
        assert compute_cer("", "text") == 1.0
        assert compute_wer("", "text") == 1.0


# ---------------------------------------------------------------------------
# Evaluate document tests
# ---------------------------------------------------------------------------


class TestEvaluateDocument:
    """Tests for document-level evaluation."""

    def test_evaluate_returns_expected_keys(self):
        """evaluate_document should return all expected keys."""
        result = evaluate_document("مرحبا", "مرحبا")
        assert "cer" in result
        assert "wer" in result
        assert "ref_chars" in result
        assert "hyp_chars" in result
        assert "ref_words" in result
        assert "hyp_words" in result

    def test_perfect_document_evaluation(self):
        """Perfect match should give cer=0, wer=0."""
        result = evaluate_document("مرحبا بالعالم", "مرحبا بالعالم")
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0


# ---------------------------------------------------------------------------
# Ground truth loader tests
# ---------------------------------------------------------------------------


class TestGroundTruthLoader:
    """Tests for the ground truth loader."""

    def test_loader_with_project_data(self):
        """Loader should find the sample images and ground truth."""
        loader = GroundTruthLoader(
            image_dir=str(project_root / "data" / "raw"),
            gt_dir=str(project_root / "data" / "ground_truth"),
        )
        pairs = loader.get_pairs()
        assert len(pairs) >= 1

    def test_loader_summary(self):
        """summary() should return valid counts."""
        loader = GroundTruthLoader(
            image_dir=str(project_root / "data" / "raw"),
            gt_dir=str(project_root / "data" / "ground_truth"),
        )
        s = loader.summary()
        assert s["total_images"] >= 2
        assert s["paired"] >= 1
        assert s["unpaired"] >= 0

    def test_loader_missing_dir_raises(self):
        """Loader should raise if image directory doesn't exist."""
        loader = GroundTruthLoader(
            image_dir="nonexistent_dir",
            gt_dir="data/ground_truth",
        )
        with pytest.raises(FileNotFoundError):
            loader.list_images()

    def test_load_ground_truth_returns_text(self):
        """load_ground_truth should return the text content."""
        loader = GroundTruthLoader(
            image_dir=str(project_root / "data" / "raw"),
            gt_dir=str(project_root / "data" / "ground_truth"),
        )
        gt = loader.load_ground_truth("sample")
        assert gt is not None
        assert len(gt) > 0

    def test_load_ground_truth_missing_returns_none(self):
        """load_ground_truth for missing file should return None."""
        loader = GroundTruthLoader(
            image_dir=str(project_root / "data" / "raw"),
            gt_dir=str(project_root / "data" / "ground_truth"),
        )
        gt = loader.load_ground_truth("nonexistent_document")
        assert gt is None
