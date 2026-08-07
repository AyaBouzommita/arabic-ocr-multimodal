import cv2
import numpy as np

class ImageEnhancer:
    """OpenCV-based image enhancement for OCR text regions."""

    def __init__(self, use_clahe: bool = True, use_denoise: bool = True):
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Clip limit 2.0 and 8x8 grid is standard for text contrast
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement and denoising to a BGR image crop.
        
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

        # 2. Denoising (removes paper grain and artifacts)
        if self.use_denoise:
            # h=10 is a good default for text. Larger h removes more noise but blurs text.
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. CLAHE Contrast Enhancement
        if self.use_clahe:
            gray = self.clahe.apply(gray)

        # 4. Optional: Otsu's Binarization (Disabled by default, CLAHE is usually enough)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to 3-channel BGR as PaddleOCR expects 3 channels
        enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return enhanced_bgr
