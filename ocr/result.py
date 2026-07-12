"""OCR result data models conforming to the Sprint 0 interface contract.

Implements the ocr_output.schema.json contract:
    {
        "document_id": "string",
        "engine": "tesseract | easyocr | paddleocr",
        "raw_text": "string",
        "tokens": [{"text": "string", "bbox": [x1,y1,x2,y2], "confidence": 0.0}],
        "processing_time_ms": 0
    }
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class OCRToken:
    """A single recognized word/token with spatial and confidence information.

    Attributes:
        text: The recognized text for this token.
        bbox: Bounding box as [x1, y1, x2, y2] in pixels.
        confidence: Engine confidence score (0.0–100.0).
    """

    text: str
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize token to a dictionary matching the interface contract."""
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class OCRResult:
    """Full OCR output for a single document, matching ocr_output.schema.json.

    Attributes:
        document_id: Unique document identifier (typically the filename stem).
        engine: OCR engine name — one of 'tesseract', 'easyocr', 'paddleocr'.
        raw_text: Full extracted text concatenated from all tokens.
        tokens: Per-word tokens with bounding box and confidence info.
        processing_time_ms: Total processing time in milliseconds.
    """

    document_id: str
    engine: str
    raw_text: str
    tokens: List[OCRToken] = field(default_factory=list)
    processing_time_ms: int = 0

    def to_dict(self) -> dict:
        """Serialize to a dictionary matching the interface contract schema."""
        return {
            "document_id": self.document_id,
            "engine": self.engine,
            "raw_text": self.raw_text,
            "tokens": [token.to_dict() for token in self.tokens],
            "processing_time_ms": self.processing_time_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string matching the interface contract schema."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "OCRResult":
        """Deserialize from a dictionary matching the interface contract schema."""
        tokens = [
            OCRToken(
                text=t["text"],
                bbox=t["bbox"],
                confidence=t["confidence"],
            )
            for t in data.get("tokens", [])
        ]
        return cls(
            document_id=data["document_id"],
            engine=data["engine"],
            raw_text=data["raw_text"],
            tokens=tokens,
            processing_time_ms=data.get("processing_time_ms", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OCRResult":
        """Deserialize from a JSON string matching the interface contract schema."""
        return cls.from_dict(json.loads(json_str))

    @property
    def avg_confidence(self) -> float:
        """Compute average confidence across all tokens."""
        if not self.tokens:
            return 0.0
        return sum(t.confidence for t in self.tokens) / len(self.tokens)

    @property
    def word_count(self) -> int:
        """Return the number of recognized tokens."""
        return len(self.tokens)

    def validate_schema(self) -> bool:
        """Validate that this result conforms to the interface contract.

        Returns:
            True if the result is valid, False otherwise.

        Raises:
            ValueError: If validation fails, with a description of the issue.
        """
        if not self.document_id:
            raise ValueError("document_id is required")
        if self.engine not in ("tesseract", "easyocr", "paddleocr"):
            raise ValueError(
                f"engine must be 'tesseract', 'easyocr', or 'paddleocr', "
                f"got '{self.engine}'"
            )
        if not isinstance(self.raw_text, str):
            raise ValueError("raw_text must be a string")
        for i, token in enumerate(self.tokens):
            if not isinstance(token.text, str):
                raise ValueError(f"tokens[{i}].text must be a string")
            if len(token.bbox) != 4:
                raise ValueError(
                    f"tokens[{i}].bbox must have exactly 4 elements, "
                    f"got {len(token.bbox)}"
                )
            if not all(isinstance(v, (int, float)) for v in token.bbox):
                raise ValueError(f"tokens[{i}].bbox values must be numeric")
        if not isinstance(self.processing_time_ms, int):
            raise ValueError("processing_time_ms must be an integer")
        return True