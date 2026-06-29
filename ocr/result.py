"""OCR result data structures — shared interface contract.

This module defines the OCRResult and OCRToken data classes used by
every OCR engine in this project (Tesseract, EasyOCR, PaddleOCR).

The structure conforms to the interface contract defined in
docs/interface_contracts.md and validated by schemas/ocr_output.schema.json.

Usage:
    from ocr.result import OCRResult, OCRToken

    token = OCRToken(text="مرحبا", bbox=[10, 20, 100, 50], confidence=95.0)
    result = OCRResult(
        document_id="doc_001",
        engine="paddleocr",
        raw_text="مرحبا",
        tokens=[token],
        processing_time_ms=120,
    )
    print(result.to_json())
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

ALLOWED_ENGINES = {"tesseract", "easyocr", "paddleocr"}


@dataclass
class OCRToken:
    """A single recognized text token with its bounding box and confidence.

    Attributes:
        text: The recognized text string.
        bbox: Bounding box as [x1, y1, x2, y2] (top-left, bottom-right).
        confidence: Confidence score in the range [0.0, 100.0].
    """

    text: str
    bbox: List[float]
    confidence: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OCRToken":
        """Deserialize from dictionary."""
        return cls(
            text=data["text"],
            bbox=data["bbox"],
            confidence=data["confidence"],
        )


@dataclass
class OCRResult:
    """Structured OCR output conforming to the shared interface contract.

    Attributes:
        document_id: Unique identifier for the processed document.
        engine: Name of the OCR engine used (tesseract, easyocr, paddleocr).
        raw_text: Full concatenated text extracted from the document.
        tokens: List of OCRToken objects with individual word data.
        processing_time_ms: Time taken to process the image in milliseconds.
    """

    document_id: str
    engine: str
    raw_text: str
    tokens: List[OCRToken] = field(default_factory=list)
    processing_time_ms: int = 0

    @property
    def word_count(self) -> int:
        """Number of tokens extracted."""
        return len(self.tokens)

    @property
    def avg_confidence(self) -> float:
        """Average confidence score across all tokens (0.0 if no tokens)."""
        if not self.tokens:
            return 0.0
        return round(sum(t.confidence for t in self.tokens) / len(self.tokens), 2)

    def validate_schema(self) -> bool:
        """Validate this result against the interface contract rules.

        Raises:
            ValueError: If any contract constraint is violated.

        Returns:
            True if valid.
        """
        if not self.document_id:
            raise ValueError("document_id must be a non-empty string")

        if self.engine not in ALLOWED_ENGINES:
            raise ValueError(
                f"engine must be one of {ALLOWED_ENGINES}, got '{self.engine}'"
            )

        for i, token in enumerate(self.tokens):
            if len(token.bbox) != 4:
                raise ValueError(
                    f"Token {i}: bbox must have exactly 4 elements [x1, y1, x2, y2], "
                    f"got {len(token.bbox)}"
                )

        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary matching the interface contract."""
        return {
            "document_id": self.document_id,
            "engine": self.engine,
            "raw_text": self.raw_text,
            "tokens": [t.to_dict() for t in self.tokens],
            "processing_time_ms": self.processing_time_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "OCRResult":
        """Deserialize from dictionary."""
        return cls(
            document_id=data["document_id"],
            engine=data["engine"],
            raw_text=data["raw_text"],
            tokens=[OCRToken.from_dict(t) for t in data.get("tokens", [])],
            processing_time_ms=data.get("processing_time_ms", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OCRResult":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
