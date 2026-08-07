import torch
from ocr.base_engine import OCREngine
from ocr.result import OCRResult, OCRToken
from ocr.tesseract.engine import TesseractEngine
from ocr.paddleocr.engine import PaddleOCREngine

try:
    from ocr.easyocr.engine import EasyOCREngine
except Exception:
    EasyOCREngine = None

__all__ = ["OCREngine", "OCRResult", "OCRToken", "TesseractEngine", "PaddleOCREngine", "EasyOCREngine"]
