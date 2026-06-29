"""Abstract base class for all OCR engines in this project.

Every engine (Tesseract, EasyOCR, PaddleOCR) must implement this interface
so they can be used interchangeably by the evaluation pipeline.

Usage:
    from ocr.base_engine import OCREngine

    class MyEngine(OCREngine):
        def extract_text(self, image_path, document_id=None):
            ...
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ocr.result import OCRResult


class OCREngine(ABC):
    """Abstract base class that all OCR engines must implement.

    Attributes:
        name: The engine identifier string (e.g., 'paddleocr').
    """

    name: str = "base"

    @abstractmethod
    def extract_text(
        self, image_path: str, document_id: Optional[str] = None
    ) -> OCRResult:
        """Extract text from a document image.

        Args:
            image_path: Absolute or relative path to the image file.
            document_id: Optional identifier for the document. Defaults to
                the image filename stem.

        Returns:
            OCRResult conforming to the shared interface contract.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be read or processed.
        """

    def _resolve_document_id(
        self, image_path: str, document_id: Optional[str]
    ) -> str:
        """Return document_id, defaulting to the filename stem.

        Args:
            image_path: Path to the image file.
            document_id: Caller-supplied identifier (may be None).

        Returns:
            The document identifier string.
        """
        if document_id:
            return document_id
        return Path(image_path).stem

    def _validate_image_path(self, image_path: str) -> Path:
        """Validate that the image file exists.

        Args:
            image_path: Path to the image file.

        Returns:
            Resolved Path object.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return path
