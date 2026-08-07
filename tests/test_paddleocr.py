"""Tests for the PaddleOCR engine.

Validates that the PaddleOCREngine:
    - Produces output matching the ocr_output.schema.json interface contract
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
from ocr.paddleocr.engine import PaddleOCREngine


# --- Fixtures ---


@pytest.fixture(scope="module")
def engine():
    """Create a PaddleOCR engine with default Arabic config.

    Uses module scope to avoid re-initializing the heavy model for each test.
    """
    return PaddleOCREngine(lang="ar")


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
        """to_json() -> from_json() must produce equivalent result."""
        result = engine.extract_text(sample_image)
        json_str = result.to_json()
        restored = OCRResult.from_json(json_str)

        assert restored.document_id == result.document_id
        assert restored.engine == result.engine
        assert restored.raw_text == result.raw_text
        assert len(restored.tokens) == len(result.tokens)


# --- OCR Quality Tests ---


class TestPaddleOCR:
    """Tests for PaddleOCR extraction quality."""

    def test_extracts_arabic_text(self, engine, sample_image):
        """PaddleOCR should extract some Arabic text from the sample."""
        result = engine.extract_text(sample_image)
        assert len(result.raw_text) > 0
        # Check that some Arabic characters are present
        arabic_chars = [c for c in result.raw_text if "\u0600" <= c <= "\u06ff"]
        assert len(arabic_chars) > 0, "No Arabic characters detected"

    def test_extracts_text_from_calligraphy(self, engine, sample2_image):
        """PaddleOCR should extract text from the calligraphy sample."""
        result = engine.extract_text(sample2_image)
        # Calligraphy may be harder -- just check it produces something
        assert isinstance(result.raw_text, str)

    def test_simple_extraction_returns_string(self, engine, sample_image):
        """extract_text_simple() should return a plain text string."""
        text = engine.extract_text_simple(sample_image)
        assert isinstance(text, str)

    def test_multiple_tokens_extracted(self, engine, sample_image):
        """Sample image should produce multiple tokens."""
        result = engine.extract_text(sample_image)
        assert len(result.tokens) >= 1

    def test_confidence_in_valid_range(self, engine, sample_image):
        """Token confidence should be between 0 and 100."""
        result = engine.extract_text(sample_image)
        for token in result.tokens:
            assert 0.0 <= token.confidence <= 100.0, (
                f"Confidence {token.confidence} out of range for '{token.text}'"
            )

    def test_bbox_coordinates_valid(self, engine, sample_image):
        """Bounding box coordinates should have x_min <= x_max and y_min <= y_max."""
        result = engine.extract_text(sample_image)
        for token in result.tokens:
            x_min, y_min, x_max, y_max = token.bbox
            assert x_min <= x_max, f"x_min ({x_min}) > x_max ({x_max})"
            assert y_min <= y_max, f"y_min ({y_min}) > y_max ({y_max})"


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


# --- Engine Configuration Tests ---


class TestEngineConfig:
    """Tests for PaddleOCR engine configuration."""

    def test_engine_name_property(self):
        """engine_name should return 'paddleocr'."""
        engine = PaddleOCREngine()
        assert engine.engine_name == "paddleocr"

    def test_lazy_initialization(self):
        """PaddleOCR model should not be loaded until first use."""
        engine = PaddleOCREngine()
        assert engine._ocr is None

    def test_custom_lang_setting(self):
        """Custom language setting should be stored."""
        engine = PaddleOCREngine(lang="en")
        assert engine.lang == "en"
