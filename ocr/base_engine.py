"""Abstract base class for all OCR engines.

Every OCR engine (Tesseract, EasyOCR, PaddleOCR) must inherit from this class
and implement the extract_text() method. The base class provides shared image
preprocessing utilities used across all engines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ocr.result import OCRResult


class OCREngine(ABC):
    """Abstract base class for OCR engines.

    Subclasses must implement extract_text() which processes an image file
    and returns an OCRResult conforming to the interface contract.
    """

    @abstractmethod
    def extract_text(self, image_path: str, document_id: Optional[str] = None) -> OCRResult:
        """Extract text from a document image.

        Args:
            image_path: Path to the input image file.
            document_id: Optional document identifier. If not provided,
                the filename stem is used.

        Returns:
            OCRResult conforming to the ocr_output.schema.json contract.
        """
        pass

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the engine identifier used in the interface contract."""
        pass

    @staticmethod
    def resolve_document_id(image_path: str, document_id: Optional[str] = None) -> str:
        """Resolve the document ID from the path or explicit argument.

        Args:
            image_path: Path to the image file.
            document_id: Optional explicit document ID.

        Returns:
            The document ID string.
        """
        if document_id:
            return document_id
        return Path(image_path).stem

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load an image from disk using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            The loaded image as a NumPy array (BGR format).

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image could not be read.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        return img

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale.

        Args:
            image: Input image (BGR or already grayscale).

        Returns:
            Grayscale image.
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_threshold(image: np.ndarray, method: str = "otsu") -> np.ndarray:
        """Apply binary thresholding to a grayscale image.

        Args:
            image: Grayscale input image.
            method: Thresholding method — 'otsu' or 'adaptive'.

        Returns:
            Binary thresholded image.
        """
        if method == "otsu":
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        return binary

    @staticmethod
    def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """Remove noise from an image using Non-Local Means Denoising.

        Args:
            image: Input image (grayscale or BGR).
            strength: Filter strength. Higher values remove more noise
                but may blur text.

        Returns:
            Denoised image.
        """
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Correct slight rotation (skew) in a document image.

        Uses the minimum area rectangle of non-zero pixels to detect
        the skew angle, then rotates to correct it.

        Args:
            image: Grayscale binary image.

        Returns:
            Deskewed image.
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 10:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def preprocess(
        self,
        image: np.ndarray,
        grayscale: bool = True,
        denoise: bool = True,
        threshold: bool = False,
        deskew: bool = False,
    ) -> np.ndarray:
        """Apply a preprocessing pipeline to a document image.

        Args:
            image: Input image (BGR).
            grayscale: Convert to grayscale.
            denoise: Apply noise removal.
            threshold: Apply binary thresholding (Otsu).
            deskew: Correct skew rotation.

        Returns:
            Preprocessed image ready for OCR.
        """
        result = image.copy()

        if grayscale:
            result = self.to_grayscale(result)

        if denoise:
            result = self.denoise(result)

        if threshold:
            result = self.apply_threshold(result)

        if deskew:
            result = self.deskew(result)

        return result