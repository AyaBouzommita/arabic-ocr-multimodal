"""Tests validating OCR output against the Sprint 0 interface contracts.

Ensures that every OCR engine produces output matching the frozen
ocr_output.schema.json contract defined in docs/interface_contracts.md.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr.result import OCRResult, OCRToken


# --- Schema Validation Tests ---


class TestOCROutputSchema:
    """Validate OCR output against the interface contract."""

    def test_valid_result_passes_schema(self):
        """A well-formed OCRResult should pass validation."""
        result = OCRResult(
            document_id="doc_001",
            engine="tesseract",
            raw_text="مرحبا",
            tokens=[
                OCRToken(text="مرحبا", bbox=[10, 20, 100, 50], confidence=95.0)
            ],
            processing_time_ms=150,
        )
        assert result.validate_schema() is True

    def test_empty_document_id_fails(self):
        """Empty document_id should fail validation."""
        result = OCRResult(
            document_id="",
            engine="tesseract",
            raw_text="text",
            processing_time_ms=100,
        )
        with pytest.raises(ValueError, match="document_id"):
            result.validate_schema()

    def test_invalid_engine_name_fails(self):
        """Engine name not in allowed set should fail validation."""
        result = OCRResult(
            document_id="doc_001",
            engine="invalid_engine",
            raw_text="text",
            processing_time_ms=100,
        )
        with pytest.raises(ValueError, match="engine must be"):
            result.validate_schema()

    def test_valid_engine_names(self):
        """All three engine names from the contract should be accepted."""
        for engine_name in ("tesseract", "easyocr", "paddleocr"):
            result = OCRResult(
                document_id="doc_001",
                engine=engine_name,
                raw_text="text",
                processing_time_ms=100,
            )
            assert result.validate_schema() is True

    def test_token_bbox_must_have_four_elements(self):
        """Token bbox with wrong number of elements should fail."""
        result = OCRResult(
            document_id="doc_001",
            engine="tesseract",
            raw_text="text",
            tokens=[
                OCRToken(text="word", bbox=[10, 20, 100], confidence=90.0)
            ],
            processing_time_ms=100,
        )
        with pytest.raises(ValueError, match="bbox must have exactly 4"):
            result.validate_schema()

    def test_to_dict_has_all_contract_keys(self):
        """to_dict() must include every key from the contract."""
        result = OCRResult(
            document_id="doc_001",
            engine="tesseract",
            raw_text="مرحبا بالعالم",
            tokens=[
                OCRToken(text="مرحبا", bbox=[10, 20, 100, 50], confidence=95.0),
                OCRToken(text="بالعالم", bbox=[110, 20, 200, 50], confidence=88.5),
            ],
            processing_time_ms=200,
        )
        d = result.to_dict()

        # Top-level keys
        assert "document_id" in d
        assert "engine" in d
        assert "raw_text" in d
        assert "tokens" in d
        assert "processing_time_ms" in d

        # Token keys
        for token in d["tokens"]:
            assert "text" in token
            assert "bbox" in token
            assert "confidence" in token
            assert len(token["bbox"]) == 4

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) should produce an equivalent result."""
        original = OCRResult(
            document_id="doc_001",
            engine="easyocr",
            raw_text="تجربة",
            tokens=[
                OCRToken(text="تجربة", bbox=[5, 10, 80, 40], confidence=92.3)
            ],
            processing_time_ms=300,
        )
        restored = OCRResult.from_dict(original.to_dict())

        assert restored.document_id == original.document_id
        assert restored.engine == original.engine
        assert restored.raw_text == original.raw_text
        assert len(restored.tokens) == len(original.tokens)
        assert restored.tokens[0].text == original.tokens[0].text
        assert restored.processing_time_ms == original.processing_time_ms
