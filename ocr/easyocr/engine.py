"""EasyOCR engine implementation for Arabic text extraction.

Implements the EasyOCREngine class that wraps the EasyOCR library to
extract Arabic text from document images. Uses reader.readtext() for
per-region bounding boxes and confidence scores, producing output
conforming to the ocr_output.schema.json interface contract.

Usage:
    engine = EasyOCREngine()
    result = engine.extract_text("path/to/document.png")
    print(result.to_json())
"""

import time
from typing import List, Optional

import easyocr

from ocr.base_engine import OCREngine
from ocr.result import OCRResult, OCRToken
from ocr.utils import sort_boxes_smart
from ocr.postprocessing.normalizer import normalize_arabic_numerals

class EasyOCREngine(OCREngine):
    """EasyOCR engine for Arabic document text extraction.

    This engine uses the EasyOCR library with Arabic language support.
    Unlike Tesseract, EasyOCR handles its own image preprocessing
    internally, so we pass the original BGR image directly.

    EasyOCR's readtext() returns results as a list of tuples:
        (bounding_box, text, confidence)
    where bounding_box is 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
    which we convert to the axis-aligned [x_min, y_min, x_max, y_max]
    format required by the interface contract.

    Attributes:
        languages: List of EasyOCR language codes (default: ['ar']).
        gpu: Whether to use GPU acceleration.
        detail: Level of detail in output (1 = full, 0 = text only).
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        paragraph: bool = False,
        detail: int = 1,
    ):
        """Initialize the EasyOCR engine.

        Args:
            languages: List of language codes. Use ['ar'] for Arabic,
                ['ar', 'en'] for Arabic + English.
            gpu: Use GPU acceleration if available.
            paragraph: Combine results into paragraphs.
            detail: Output detail level. 1 = full (bbox + confidence),
                0 = text only.
        """
        self.languages = languages or ["ar"]
        self.gpu = gpu
        self.paragraph = paragraph
        self.detail = detail

        # Initialize the EasyOCR reader (lazy-loaded on first use)
        self._reader = None

    def _get_reader(self) -> easyocr.Reader:
        """Lazy-initialize the EasyOCR reader.

        The reader downloads model files on first use, so we defer
        initialization until the first call to extract_text().

        Returns:
            Initialized EasyOCR Reader instance.
        """
        if self._reader is None:
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
            )
        return self._reader

    @property
    def engine_name(self) -> str:
        """Return the engine identifier for the interface contract."""
        return "easyocr"

    def extract_text(
        self,
        image_path: str,
        document_id: Optional[str] = None,
    ) -> OCRResult:
        """Extract Arabic text from a document image using EasyOCR.

        Runs the full pipeline: load image -> EasyOCR readtext() ->
        parse per-region data -> build OCRResult matching the interface
        contract.

        Note: EasyOCR handles its own preprocessing internally, so we
        pass the BGR image directly without grayscale conversion.

        Args:
            image_path: Path to the input image file.
            document_id: Optional document identifier. Defaults to filename stem.

        Returns:
            OCRResult with raw_text, per-region tokens (bbox + confidence),
            and processing time in milliseconds.
        """
        doc_id = self.resolve_document_id(image_path, document_id)

        start_time = time.perf_counter()

        # Load the image (BGR format) — EasyOCR handles its own
        # preprocessing, so no grayscale/threshold conversion.
        image = self.load_image(image_path)

        # Run EasyOCR with tuned parameters for better symbol detection
        reader = self._get_reader()
        easyocr_results = reader.readtext(
            image,
            detail=self.detail,
            paragraph=self.paragraph,
            text_threshold=0.5,  # Lowered from 0.7 to catch small punctuation
            mag_ratio=1.5,       # Magnify to help with tiny symbols
        )

        # Parse tokens from EasyOCR output
        tokens = []
        text_parts = []

        # Sort tokens Right-to-Left for Arabic layout support
        def get_item_bbox(item):
            box_p = item[0]
            return [min(p[0] for p in box_p), min(p[1] for p in box_p),
                    max(p[0] for p in box_p), max(p[1] for p in box_p)]
                    
        # Determine direction based on loaded languages
        is_rtl = 'ar' in self.languages
        easyocr_results = sort_boxes_smart(easyocr_results, get_item_bbox, is_rtl=is_rtl)

        # EasyOCR returns list of (bbox, text, confidence)
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        for result_item in easyocr_results:
            box_points = result_item[0]  # 4-point polygon
            text = result_item[1]        # recognized text
            text = normalize_arabic_numerals(text)
            
            # paragraph=True doesn't return confidence, default to 0.8 to allow corrections
            conf = result_item[2] if len(result_item) > 2 else 0.8

            if not text.strip():
                continue

            # Convert 4-point polygon to axis-aligned bbox
            # [x_min, y_min, x_max, y_max]
            xs = [p[0] for p in box_points]
            ys = [p[1] for p in box_points]
            bbox = [
                float(min(xs)),
                float(min(ys)),
                float(max(xs)),
                float(max(ys)),
            ]

            # EasyOCR confidence is 0.0-1.0, convert to 0-100 scale
            # to match Tesseract's convention used in the project
            confidence = float(conf) * 100.0

            token = OCRToken(
                text=text.strip(),
                bbox=bbox,
                confidence=confidence,
            )
            tokens.append(token)
            text_parts.append(text.strip())

        # Build raw text — join with spaces (Arabic reads RTL but
        # string storage is logical order)
        raw_text = " ".join(text_parts)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        result = OCRResult(
            document_id=doc_id,
            engine=self.engine_name,
            raw_text=raw_text,
            tokens=tokens,
            processing_time_ms=elapsed_ms,
        )

        return result

    def extract_text_simple(self, image_path: str) -> str:
        """Quick extraction returning only the raw text string.

        Useful for quick tests and comparisons without the full
        structured output.

        Args:
            image_path: Path to the input image file.

        Returns:
            The extracted raw text as a string.
        """
        image = self.load_image(image_path)

        reader = self._get_reader()
        easyocr_results = reader.readtext(
            image,
            detail=self.detail,
            paragraph=self.paragraph,
        )

        text_parts = []
        for result_item in easyocr_results:
            text = result_item[1]
            if text.strip():
                text_parts.append(text.strip())

        return " ".join(text_parts)
