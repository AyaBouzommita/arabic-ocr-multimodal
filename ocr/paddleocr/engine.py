"""PaddleOCR engine implementation for Arabic text extraction.

Implements the PaddleOCREngine class that wraps PaddlePaddle's PaddleOCR
to extract Arabic text from document images. Uses the PaddleOCR API for
per-line bounding boxes and confidence scores, producing output conforming
to the ocr_output.schema.json interface contract.

Usage:
    engine = PaddleOCREngine()
    result = engine.extract_text("path/to/document.png")
    print(result.to_json())
"""

import time
from typing import Optional

from paddleocr import PaddleOCR

from ocr.base_engine import OCREngine
from ocr.result import OCRResult, OCRToken


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine for Arabic document text extraction.

    This engine uses PaddlePaddle's PaddleOCR with Arabic language support.
    Unlike the Tesseract engine, PaddleOCR handles its own image
    preprocessing internally (normalization, resizing), so we pass the
    original BGR image directly without grayscale/threshold conversion.

    PaddleOCR returns results as a list of pages, where each page is a
    list of lines. Each line is [bounding_box_points, (text, confidence)].
    The bounding box is given as 4 corner points
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], which we convert to the
    axis-aligned [x_min, y_min, x_max, y_max] format required by the
    interface contract.

    Attributes:
        lang: PaddleOCR language code (default: 'ar' for Arabic).
        use_angle_cls: Whether to use text angle classification.
        use_gpu: Whether to use GPU acceleration.
        det_db_thresh: DB detection threshold.
        det_db_box_thresh: DB box detection threshold.
    """

    def __init__(
        self,
        lang: str = "ar",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
    ):
        """Initialize the PaddleOCR engine.

        Args:
            lang: PaddleOCR language code. Use 'ar' for Arabic.
            use_angle_cls: Enable text angle classification for rotated text.
            use_gpu: Use GPU acceleration if available.
            det_db_thresh: DB detection threshold (lower = more sensitive).
            det_db_box_thresh: DB box detection threshold.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh

        # Initialize the PaddleOCR model (lazy-loaded on first use)
        self._ocr = None

    def _get_ocr(self) -> PaddleOCR:
        """Lazy-initialize the PaddleOCR model.

        The model is heavy and downloads data on first use, so we
        defer initialization until the first call to extract_text().

        Returns:
            Initialized PaddleOCR instance.
        """
        if self._ocr is None:
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                use_gpu=self.use_gpu,
                det_db_thresh=self.det_db_thresh,
                det_db_box_thresh=self.det_db_box_thresh,
                show_log=False,
            )
        return self._ocr

    @property
    def engine_name(self) -> str:
        """Return the engine identifier for the interface contract."""
        return "paddleocr"

    def extract_text(
        self,
        image_path: str,
        document_id: Optional[str] = None,
    ) -> OCRResult:
        """Extract Arabic text from a document image using PaddleOCR.

        Runs the full pipeline: load image -> PaddleOCR -> parse per-line
        data -> build OCRResult matching the interface contract.

        Note: Unlike TesseractEngine, we do NOT apply grayscale/threshold
        preprocessing because PaddleOCR's internal pipeline expects BGR
        images and handles its own normalization.

        Args:
            image_path: Path to the input image file.
            document_id: Optional document identifier. Defaults to filename stem.

        Returns:
            OCRResult with raw_text, per-line tokens (bbox + confidence),
            and processing time in milliseconds.
        """
        doc_id = self.resolve_document_id(image_path, document_id)

        start_time = time.perf_counter()

        # Load the image (BGR format) — do NOT convert to grayscale,
        # PaddleOCR needs 3-channel input for its internal normalization.
        image = self.load_image(image_path)

        # Run PaddleOCR
        ocr = self._get_ocr()
        paddle_results = ocr.ocr(image, cls=self.use_angle_cls)

        # Parse tokens from PaddleOCR output
        tokens = []
        text_parts = []

        # PaddleOCR returns a list of pages; each page is a list of lines
        # Each line is [bounding_box_points, (text, confidence)]
        if paddle_results and paddle_results[0]:
            for line in paddle_results[0]:
                box_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = line[1][0]     # recognized text
                conf = line[1][1]     # confidence score (0.0-1.0)

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

                # PaddleOCR confidence is 0.0-1.0, convert to 0-100 scale
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

        ocr = self._get_ocr()
        paddle_results = ocr.ocr(image, cls=self.use_angle_cls)

        text_parts = []
        if paddle_results and paddle_results[0]:
            for line in paddle_results[0]:
                text = line[1][0]
                if text.strip():
                    text_parts.append(text.strip())

        return " ".join(text_parts)
