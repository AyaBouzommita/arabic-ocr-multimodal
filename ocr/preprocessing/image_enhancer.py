"""Advanced image preprocessing for OCR text regions.

Provides multi-stage enhancement pipeline:
1. Bilateral filtering (edge-preserving denoise)
2. CLAHE contrast enhancement
3. Sauvola adaptive binarization
4. Morphological cleanup
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


class ImageEnhancer:
    """Advanced OpenCV + scikit-image enhancement for OCR text regions."""

    def __init__(
        self,
        use_clahe: bool = True,
        use_denoise: bool = True,
        use_binarization: bool = False,
        use_bilateral: bool = True,
    ):
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        self.use_binarization = use_binarization
        self.use_bilateral = use_bilateral

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-stage enhancement to a BGR image crop.

        Pipeline order:
        1. Bilateral filter (preserves edges while removing noise)
        2. CLAHE contrast enhancement
        3. Optional: Sauvola adaptive binarization
        4. Optional: Morphological cleanup

        Args:
            image: BGR numpy array of the cropped text region.

        Returns:
            Enhanced BGR numpy array (3 channels).
        """
        if image is None or image.size == 0:
            return image

        # 1. Convert to Grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Bilateral filter (edge-preserving denoising)
        #    - Better than fastNlMeansDenoising for text: preserves stroke edges
        #    - d=7: neighborhood diameter
        #    - sigmaColor=50: color space filter sigma
        #    - sigmaSpace=50: coordinate space filter sigma
        if self.use_bilateral:
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        elif self.use_denoise:
            gray = cv2.fastNlMeansDenoising(
                gray, h=10, templateWindowSize=7, searchWindowSize=21
            )

        # 3. CLAHE Contrast Enhancement
        if self.use_clahe:
            gray = self.clahe.apply(gray)

        # 4. Sauvola adaptive binarization (superior to Otsu for documents)
        #    Calculates local threshold based on mean + std deviation
        if self.use_binarization:
            sauvola_thresh = threshold_sauvola(gray, window_size=25, k=0.2)
            binary = (gray > sauvola_thresh).astype(np.uint8) * 255

            # 5. Morphological cleanup (remove tiny noise specks)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            gray = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Convert back to 3-channel BGR (OCR engines expect 3 channels)
        enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return enhanced_bgr

    def enhance_crop_arabic(self, image: np.ndarray, target_height: int = 128) -> np.ndarray:
        """Apply Arabic-optimized enhancement to a BGR image crop.

        Arabic script has many small dots, diacritics, and complex ligatures
        that get lost in small crops. This pipeline:
        1. Upscales small crops so EasyOCR can distinguish fine details
        2. Applies bilateral filtering to clean noise while preserving edges
        3. Uses CLAHE for contrast enhancement
        4. Applies unsharp masking to sharpen strokes and dots

        Only used for the EasyOCR/Arabic branch. Does NOT affect PaddleOCR/French.

        Args:
            image: BGR numpy array of the cropped text region.
            target_height: Minimum height in pixels. Crops shorter than this
                will be upscaled proportionally.

        Returns:
            Enhanced BGR numpy array (3 channels).
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]

        # 1. Upscale small crops (Arabic dots/diacritics need at least ~128px height)
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 3. Bilateral filter (edge-preserving denoising — gentler settings for Arabic)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 4. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 5. Unsharp masking — sharpens stroke edges and dots
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # Convert back to 3-channel BGR
        enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return enhanced_bgr
