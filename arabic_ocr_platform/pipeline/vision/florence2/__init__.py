"""Florence-2 detection candidate for the vision bake-off."""

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.engine import Florence2Detector
from arabic_ocr_platform.pipeline.vision.florence2.evaluator import evaluate_florence2
from arabic_ocr_platform.pipeline.vision.florence2.trainer import Florence2Trainer

__all__ = [
    "Florence2Config",
    "Florence2Detector",
    "Florence2Trainer",
    "evaluate_florence2",
]
