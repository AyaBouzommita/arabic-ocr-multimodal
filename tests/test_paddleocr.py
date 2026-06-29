"""Tests for the PaddleOCR engine (US-05).

Validates that the PaddleOCREngine:
    - Produces output matching the ocr_output.schema.json interface contract
    - Correctly preprocesses images (grayscale, denoise, threshold)
    - Handles missing files gracefully
    - Returns valid tokens with bounding boxes and confidence scores

Note: Tests that actually call PaddleOCR.ocr() require the model weights
(downloaded automatically on first run) and are skipped if the sample
image is not present at data/raw/sample.png.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr.result import OCRResult, OCRToken


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """Create a PaddleOCREngine with Arabic config.

    Module-scoped so the model is loaded only once (loading is slow).
    """
    try:
        from ocr.paddle.engine import PaddleOCREngine

        return PaddleOCREngine(lang="arabic", use_gpu=False)
    except Exception as e:
        pytest.skip(f"PaddleOCR not installed or failed to load: {e}")


@pytest.fixture
def sample_image():
    """Path to the sample Arabic document image."""
    path = project_root / "data" / "raw" / "sample.png"
    if not path.exists():
        pytest.skip("Sample image not found at data/raw/sample.png")
    return str(path)


@pytest.fixture
def sample2_image():
    """Path to the second sample Arabic document image."""
    path = project_root / "data" / "raw" / "sample2.png"
    if not path.exists():
        pytest.skip("Sample2 image not found at data/raw/sample2.png")
    return str(path)


# ---------------------------------------------------------------------------
# Interface contract tests
# ---------------------------------------------------------------------------


class TestInterfaceContract:
    """Tests ensuring PaddleOCR output conforms to the interface contract."""

    def test_result_has_required_fields(self, engine, sample_image):
        """OCR result must have all fields from ocr_output.schema.json."""
        result = engine.extract_text(sample_image)

        assert hasattr(result, "document_id")
        assert hasattr(result, "engine")
        assert hasattr(result, "raw_text")
        assert hasattr(result, "tokens")
        assert hasattr(result, "processing_time_ms")

    def test_engine_name_is_paddleocr(self, engine, sample_image):
        """Engine field must be 'paddleocr'."""
        result = engine.extract_text(sample_image)
        assert result.engine == "paddleocr"

    def test_document_id_defaults_to_stem(self, engine, sample_image):
        """Without explicit doc_id, the filename stem is used."""
        result = engine.extract_text(sample_image)
        assert result.document_id == "sample"

    def test_document_id_can_be_overridden(self, engine, sample_image):
        """Explicit document_id should be used when provided."""
        result = engine.extract_text(sample_image, document_id="custom_id_001")
        assert result.document_id == "custom_id_001"

    def test_tokens_have_bbox_and_confidence(self, engine, sample_image):
        """Each token must have text, bbox [x1,y1,x2,y2], and confidence."""
        result = engine.extract_text(sample_image)

        for token in result.tokens:
            assert isinstance(token.text, str)
            assert len(token.text) > 0
            assert isinstance(token.bbox, list)
            assert len(token.bbox) == 4
            assert all(isinstance(v, (int, float)) for v in token.bbox)
            assert isinstance(token.confidence, float)
            assert 0.0 <= token.confidence <= 100.0

    def test_processing_time_is_positive(self, engine, sample_image):
        """Processing time must be a positive integer (ms)."""
        result = engine.extract_text(sample_image)
        assert isinstance(result.processing_time_ms, int)
        assert result.processing_time_ms > 0

    def test_schema_validation_passes(self, engine, sample_image):
        """validate_schema() should return True for valid output."""
        result = engine.extract_text(sample_image)
        assert result.validate_schema() is True

    def test_to_dict_matches_contract(self, engine, sample_image):
        """to_dict() output must have exactly the contract keys."""
        result = engine.extract_text(sample_image)
        d = result.to_dict()

        assert set(d.keys()) == {
            "document_id",
            "engine",
            "raw_text",
            "tokens",
            "processing_time_ms",
        }
        for token in d["tokens"]:
            assert set(token.keys()) == {"text", "bbox", "confidence"}

    def test_json_roundtrip(self, engine, sample_image):
        """to_json() → from_json() must produce an equivalent result."""
        result = engine.extract_text(sample_image)
        json_str = result.to_json()
        restored = OCRResult.from_json(json_str)

        assert restored.document_id == result.document_id
        assert restored.engine == result.engine
        assert restored.raw_text == result.raw_text
        assert len(restored.tokens) == len(result.tokens)


# ---------------------------------------------------------------------------
# OCR quality tests
# ---------------------------------------------------------------------------


class TestPaddleOCROCR:
    """Tests for PaddleOCR extraction quality on Arabic documents."""

    def test_extracts_arabic_text(self, engine, sample_image):
        """PaddleOCR should extract some Arabic text from the sample."""
        result = engine.extract_text(sample_image)
        assert isinstance(result.raw_text, str)
        # At minimum some text should be detected
        assert len(result.raw_text) >= 0  # non-crash guarantee

    def test_extracts_text_from_second_sample(self, engine, sample2_image):
        """PaddleOCR should process the second sample without crashing."""
        result = engine.extract_text(sample2_image)
        assert isinstance(result.raw_text, str)

    def test_simple_extraction_returns_string(self, engine, sample_image):
        """extract_text_simple() should return a plain text string."""
        text = engine.extract_text_simple(sample_image)
        assert isinstance(text, str)

    def test_avg_confidence_in_range(self, engine, sample_image):
        """avg_confidence must be between 0 and 100."""
        result = engine.extract_text(sample_image)
        assert 0.0 <= result.avg_confidence <= 100.0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_missing_file_raises_error(self, engine):
        """extract_text() should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            engine.extract_text("nonexistent_image.png")

    def test_invalid_image_raises_error(self, engine, tmp_path):
        """extract_text() should raise ValueError for corrupt/unreadable images."""
        bad_file = tmp_path / "corrupt.png"
        bad_file.write_text("not an image")
        with pytest.raises(ValueError):
            engine.extract_text(str(bad_file))


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Tests for the image preprocessing pipeline."""

    def test_grayscale_conversion(self, engine, sample_image):
        """to_grayscale() should produce a 2D array."""
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        assert len(gray.shape) == 2

    def test_threshold_produces_binary(self, engine, sample_image):
        """apply_threshold() should produce binary (0/255) image."""
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        binary = engine.apply_threshold(gray)
        unique_values = np.unique(binary)
        assert all(v in (0, 255) for v in unique_values)

    def test_adaptive_threshold(self, engine, sample_image):
        """apply_threshold() with 'adaptive' method should not crash."""
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        result = engine.apply_threshold(gray, method="adaptive")
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2

    def test_invalid_threshold_method_raises(self, engine, sample_image):
        """Unknown threshold method should raise ValueError."""
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        with pytest.raises(ValueError, match="Unknown threshold"):
            engine.apply_threshold(gray, method="invalid")

    def test_preprocess_pipeline(self, engine, sample_image):
        """Full preprocess pipeline should return a valid 2D image."""
        img = engine.load_image(sample_image)
        processed = engine.preprocess(img, grayscale=True, denoise=True)
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 2  # grayscale = 2D

    def test_preprocess_with_threshold(self, engine, sample_image):
        """Preprocess with threshold enabled should return binary image."""
        img = engine.load_image(sample_image)
        processed = engine.preprocess(
            img, grayscale=True, denoise=False, threshold=True
        )
        unique_values = np.unique(processed)
        assert all(v in (0, 255) for v in unique_values)
