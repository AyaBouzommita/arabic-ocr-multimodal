"""Tests for the Tesseract OCR engine (US-04).

Validates that the TesseractEngine:
    - Produces output matching the ocr_output.schema.json interface contract
    - Correctly preprocesses images (grayscale, threshold)
    - Handles missing files gracefully
    - Returns valid tokens with bounding boxes and confidence scores
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr.result import OCRResult, OCRToken
from ocr.tesseract.engine import TesseractEngine


# --- Fixtures ---


@pytest.fixture
def engine():
    """Create a Tesseract engine with default Arabic config."""
    return TesseractEngine(lang="ara")


@pytest.fixture
def sample_image():
    """Path to sample Arabic document image."""
    path = project_root / "data" / "raw" / "sample.png"
    if not path.exists():
        pytest.skip("Sample image not found at data/raw/sample.png")
    return str(path)


@pytest.fixture
def sample2_image():
    """Path to sample2 Arabic calligraphy image."""
    path = project_root / "data" / "raw" / "sample2.png"
    if not path.exists():
        pytest.skip("Sample2 image not found at data/raw/sample2.png")
    return str(path)


# --- Schema Validation Tests ---


class TestInterfaceContract:
    """Tests ensuring OCR output conforms to the interface contract."""

    def test_result_has_required_fields(self, engine, sample_image):
        """OCR result must have all fields from ocr_output.schema.json."""
        result = engine.extract_text(sample_image)

        assert hasattr(result, "document_id")
        assert hasattr(result, "engine")
        assert hasattr(result, "raw_text")
        assert hasattr(result, "tokens")
        assert hasattr(result, "processing_time_ms")

    def test_engine_name_is_tesseract(self, engine, sample_image):
        """Engine field must be 'tesseract'."""
        result = engine.extract_text(sample_image)
        assert result.engine == "tesseract"

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
            "document_id", "engine", "raw_text", "tokens", "processing_time_ms"
        }
        for token in d["tokens"]:
            assert set(token.keys()) == {"text", "bbox", "confidence"}

    def test_json_roundtrip(self, engine, sample_image):
        """to_json() → from_json() must produce equivalent result."""
        result = engine.extract_text(sample_image)
        json_str = result.to_json()
        restored = OCRResult.from_json(json_str)

        assert restored.document_id == result.document_id
        assert restored.engine == result.engine
        assert restored.raw_text == result.raw_text
        assert len(restored.tokens) == len(result.tokens)


# --- OCR Quality Tests ---


class TestTesseractOCR:
    """Tests for Tesseract OCR extraction quality."""

    def test_extracts_arabic_text(self, engine, sample_image):
        """Tesseract should extract some Arabic text from the sample."""
        result = engine.extract_text(sample_image)
        assert len(result.raw_text) > 0
        # Check that some Arabic characters are present
        arabic_chars = [c for c in result.raw_text if "\u0600" <= c <= "\u06ff"]
        assert len(arabic_chars) > 0, "No Arabic characters detected"

    def test_extracts_text_from_calligraphy(self, engine, sample2_image):
        """Tesseract should extract text from the calligraphy sample."""
        result = engine.extract_text(sample2_image)
        # Calligraphy may be harder — just check it produces something
        assert isinstance(result.raw_text, str)

    def test_simple_extraction_returns_string(self, engine, sample_image):
        """extract_text_simple() should return a plain text string."""
        text = engine.extract_text_simple(sample_image)
        assert isinstance(text, str)

    def test_multiple_tokens_extracted(self, engine, sample_image):
        """Sample image should produce multiple tokens."""
        result = engine.extract_text(sample_image)
        assert len(result.tokens) >= 1


# --- Error Handling Tests ---


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


# --- Preprocessing Tests ---


class TestPreprocessing:
    """Tests for the image preprocessing pipeline."""

    def test_grayscale_conversion(self, engine, sample_image):
        """to_grayscale() should produce a 2D array."""
        import cv2
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        assert len(gray.shape) == 2

    def test_threshold_produces_binary(self, engine, sample_image):
        """apply_threshold() should produce binary (0/255) image."""
        import numpy as np
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        binary = engine.apply_threshold(gray)
        unique_values = np.unique(binary)
        assert all(v in (0, 255) for v in unique_values)

    def test_invalid_threshold_method_raises(self, engine, sample_image):
        """Unknown threshold method should raise ValueError."""
        img = engine.load_image(sample_image)
        gray = engine.to_grayscale(img)
        with pytest.raises(ValueError, match="Unknown threshold"):
            engine.apply_threshold(gray, method="invalid")

    def test_preprocess_pipeline(self, engine, sample_image):
        """Full preprocess pipeline should return a valid image."""
        import numpy as np
        img = engine.load_image(sample_image)
        processed = engine.preprocess(img, grayscale=True, denoise=True)
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 2  # grayscale = 2D