"""PaddleOCR Arabic engine — wraps PaddleOCR for Arabic document OCR.

This module is the primary deliverable for US-05:
    'As the assigned engineer, I implement PaddleOCR Arabic baseline
     and measure CER/WER'

It provides the PaddleOCREngine class that:
    - Loads the PaddleOCR model configured for Arabic (lang='arabic')
    - Preprocesses images (grayscale, optional denoising and thresholding)
    - Runs PaddleOCR inference and extracts text with bounding boxes
    - Returns output conforming to the shared OCRResult interface contract
    - Saves results as JSON for the shared CER/WER evaluation pipeline

Usage:
    from ocr.paddle.engine import PaddleOCREngine

    engine = PaddleOCREngine()
    result = engine.extract_text("data/raw/sample.png")
    print(result.to_json())
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ocr.base_engine import OCREngine
from ocr.result import OCRResult, OCRToken


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine configured for Arabic document recognition.

    PaddleOCR uses a detection → recognition pipeline. For Arabic, we use:
        - lang='arabic'  : loads the Arabic recognition model
        - use_angle_cls=True : enables text orientation classification,
          important for scanned documents that may be rotated
        - use_gpu=False  : CPU inference for compatibility (set True if CUDA)

    Attributes:
        name: Engine identifier, always 'paddleocr'.
        lang: PaddleOCR language code (default 'arabic').
        use_angle_cls: Whether to enable angle classification.
        enable_denoise: Whether to apply denoising in preprocessing.
        enable_threshold: Whether to apply Otsu binarisation.
    """

    name = "paddleocr"

    def __init__(
        self,
        lang: str = "arabic",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        enable_denoise: bool = True,
        enable_threshold: bool = False,
    ):
        """Initialise the PaddleOCR engine and load the model.

        Args:
            lang: PaddleOCR language code. Use 'arabic' for Arabic documents.
            use_angle_cls: Enable text angle classification (recommended for
                scanned documents). Slightly slower but more robust.
            use_gpu: Set True to run on GPU (requires CUDA-enabled PaddlePaddle).
            enable_denoise: Apply cv2 fast NL means denoising before OCR.
            enable_threshold: Apply Otsu binarisation before OCR.
        """
        from paddleocr import PaddleOCR  # deferred import — slow to load

        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.enable_denoise = enable_denoise
        self.enable_threshold = enable_threshold

        self._ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            show_log=False,  # suppress verbose paddle logging
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_text(
        self, image_path: str, document_id: Optional[str] = None
    ) -> OCRResult:
        """Run PaddleOCR on an image and return a structured OCRResult.

        Args:
            image_path: Path to the image file to process.
            document_id: Optional document identifier. Defaults to the
                image filename stem (e.g., 'sample' for 'sample.png').

        Returns:
            OCRResult conforming to the shared interface contract.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be read by OpenCV.
        """
        self._validate_image_path(image_path)
        doc_id = self._resolve_document_id(image_path, document_id)

        img = self.load_image(image_path)
        processed = self.preprocess(
            img,
            grayscale=True,
            denoise=self.enable_denoise,
            threshold=self.enable_threshold,
        )

        start = time.perf_counter()
        raw_paddle_result = self._ocr.ocr(processed, cls=self.use_angle_cls)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        tokens = self._parse_paddle_output(raw_paddle_result)
        raw_text = " ".join(t.text for t in tokens if t.text.strip())

        return OCRResult(
            document_id=doc_id,
            engine=self.name,
            raw_text=raw_text,
            tokens=tokens,
            processing_time_ms=elapsed_ms,
        )

    def extract_text_simple(self, image_path: str) -> str:
        """Return only the raw text string (no JSON metadata).

        Convenience method for quick command-line tests.

        Args:
            image_path: Path to the image file.

        Returns:
            The extracted text as a plain string.
        """
        result = self.extract_text(image_path)
        return result.raw_text

    # ------------------------------------------------------------------
    # Image loading & preprocessing
    # ------------------------------------------------------------------

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as a NumPy array (BGR colour).

        Raises:
            ValueError: If OpenCV cannot decode the file.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(
                f"OpenCV could not read image (corrupt or unsupported format): "
                f"{image_path}"
            )
        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale.

        Args:
            img: BGR image array.

        Returns:
            2-D grayscale image array.
        """
        if len(img.shape) == 2:
            return img  # already grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_threshold(
        self, gray: np.ndarray, method: str = "otsu"
    ) -> np.ndarray:
        """Apply binarisation to a grayscale image.

        Args:
            gray: 2-D grayscale image.
            method: Thresholding method. Supported: 'otsu', 'adaptive'.

        Returns:
            Binary image (0/255).

        Raises:
            ValueError: If an unknown method is specified.
        """
        if method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary
        elif method == "adaptive":
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=31,
                C=15,
            )
        else:
            raise ValueError(
                f"Unknown threshold method: '{method}'. "
                f"Supported: 'otsu', 'adaptive'."
            )

    def preprocess(
        self,
        img: np.ndarray,
        grayscale: bool = True,
        denoise: bool = True,
        threshold: bool = False,
    ) -> np.ndarray:
        """Full preprocessing pipeline before passing image to PaddleOCR.

        Steps (in order):
            1. Grayscale conversion (if enabled)
            2. Fast NL-means denoising (if enabled)
            3. Otsu binarisation (if enabled)

        Args:
            img: Input image (BGR or grayscale NumPy array).
            grayscale: Convert to grayscale before processing.
            denoise: Apply cv2 fastNlMeansDenoising.
            threshold: Apply Otsu binarisation.

        Returns:
            Preprocessed image as NumPy array.
        """
        if grayscale:
            img = self.to_grayscale(img)
        if denoise:
            img = cv2.fastNlMeansDenoising(img, h=10)
        if threshold:
            img = self.apply_threshold(img, method="otsu")
        return img

    # ------------------------------------------------------------------
    # PaddleOCR output parsing
    # ------------------------------------------------------------------

    def _parse_paddle_output(self, paddle_result) -> List[OCRToken]:
        """Convert raw PaddleOCR output to a list of OCRToken objects.

        PaddleOCR returns a nested list:
            [ [ [bbox_points, (text, confidence)], ... ] ]

        where bbox_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (polygon),
        and confidence is a float in [0.0, 1.0].

        We convert the polygon to an axis-aligned bounding box [x1,y1,x2,y2]
        and scale confidence to [0.0, 100.0] to match the interface contract.

        Args:
            paddle_result: Raw output from PaddleOCR.ocr().

        Returns:
            List of OCRToken objects. Empty list if no text detected.
        """
        tokens = []

        if not paddle_result or paddle_result == [None]:
            return tokens

        for page_result in paddle_result:
            if page_result is None:
                continue
            for line in page_result:
                bbox_points, (text, confidence) = line

                # Convert polygon [[x,y], ...] → axis-aligned bbox [x1,y1,x2,y2]
                xs = [pt[0] for pt in bbox_points]
                ys = [pt[1] for pt in bbox_points]
                x1, y1 = float(min(xs)), float(min(ys))
                x2, y2 = float(max(xs)), float(max(ys))

                tokens.append(
                    OCRToken(
                        text=text,
                        bbox=[x1, y1, x2, y2],
                        confidence=round(float(confidence) * 100, 2),
                    )
                )

        return tokens
