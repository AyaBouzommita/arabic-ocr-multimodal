"""Tesseract OCR engine implementation for Arabic text extraction.

Implements the TesseractEngine class that wraps pytesseract to extract
Arabic text from document images. Uses image_to_data() for per-word
bounding boxes and confidence scores, producing output conforming to
the ocr_output.schema.json interface contract.

Usage:
    engine = TesseractEngine()
    result = engine.extract_text("path/to/document.png")
    print(result.to_json())
"""

import os
import time
from typing import Optional

import pytesseract
from PIL import Image

from ocr.base_engine import OCREngine
from ocr.result import OCRResult, OCRToken


# Configure Tesseract executable path from environment or default
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class TesseractEngine(OCREngine):
    """Tesseract OCR engine for Arabic document text extraction.

    This engine uses Google's Tesseract OCR with Arabic language support.
    It applies image preprocessing (grayscale, denoising) before running
    OCR, and extracts per-word tokens with bounding boxes and confidence
    scores via pytesseract.image_to_data().

    Attributes:
        lang: Tesseract language code (default: 'ara' for Arabic).
        psm: Page Segmentation Mode (default: 3 — fully automatic).
        oem: OCR Engine Mode (default: 3 — LSTM + legacy).
        preprocess_config: Dict controlling which preprocessing steps to apply.
    """

    def __init__(
        self,
        lang: str = "ara",
        psm: int = 3,
        oem: int = 3,
        enable_grayscale: bool = True,
        enable_denoise: bool = True,
        enable_threshold: bool = False,
        enable_deskew: bool = False,
    ):
        """Initialize the Tesseract engine.

        Args:
            lang: Tesseract language code. Use 'ara' for Arabic,
                'ara+eng' for Arabic + English.
            psm: Page Segmentation Mode.
                3 = Fully automatic page segmentation (default).
                6 = Assume a single uniform block of text.
                11 = Sparse text — find as much text as possible.
            oem: OCR Engine Mode.
                0 = Legacy engine only.
                1 = Neural nets LSTM engine only.
                3 = Default, based on what is available.
            enable_grayscale: Apply grayscale conversion.
            enable_denoise: Apply noise removal.
            enable_threshold: Apply Otsu binary thresholding.
            enable_deskew: Apply skew correction.
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self._preprocess_config = {
            "grayscale": enable_grayscale,
            "denoise": enable_denoise,
            "threshold": enable_threshold,
            "deskew": enable_deskew,
        }

    @property
    def engine_name(self) -> str:
        """Return the engine identifier for the interface contract."""
        return "tesseract"

    @property
    def tesseract_config(self) -> str:
        """Build the Tesseract CLI config string.

        Returns:
            Config string with PSM and OEM flags.
        """
        return f"--psm {self.psm} --oem {self.oem}"

    def extract_text(
        self,
        image_path: str,
        document_id: Optional[str] = None,
    ) -> OCRResult:
        """Extract Arabic text from a document image using Tesseract.

        Runs the full pipeline: load image → preprocess → Tesseract OCR →
        parse per-word data → build OCRResult matching the interface contract.

        Args:
            image_path: Path to the input image file.
            document_id: Optional document identifier. Defaults to filename stem.

        Returns:
            OCRResult with raw_text, per-word tokens (bbox + confidence),
            and processing time in milliseconds.
        """
        doc_id = self.resolve_document_id(image_path, document_id)

        start_time = time.perf_counter()

        # Load and preprocess the image
        image = self.load_image(image_path)
        preprocessed = self.preprocess(image, **self._preprocess_config)

        # Convert to PIL for pytesseract
        pil_image = Image.fromarray(preprocessed)

        # Extract per-word data using image_to_data for bboxes + confidence
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT,
        )

        # Parse tokens from Tesseract data output
        tokens = []
        text_parts = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])

            # Skip empty tokens and those with -1 confidence (non-text)
            if not word or conf < 0:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            token = OCRToken(
                text=word,
                bbox=[x, y, x + w, y + h],
                confidence=conf,
            )
            tokens.append(token)
            text_parts.append(word)

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
        preprocessed = self.preprocess(image, **self._preprocess_config)
        pil_image = Image.fromarray(preprocessed)

        text = pytesseract.image_to_string(
            pil_image,
            lang=self.lang,
            config=self.tesseract_config,
        )

        return text.strip()
