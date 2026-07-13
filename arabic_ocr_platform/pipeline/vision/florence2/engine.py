"""Florence-2 detection engine for inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config


class Florence2Detector:
    """Wrapper around Florence-2 for object-detection inference."""

    def __init__(self, config: Optional[Florence2Config] = None, model_dir: Optional[Path] = None):
        self.config = config or Florence2Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = str(model_dir) if model_dir else self.config.checkpoint
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            revision=self.config.revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            revision=self.config.revision,
        ).to(self.device)
        self.model.eval()

    def predict_image(
        self,
        image: Image.Image,
        allowed_classes: Optional[Set[str]] = None,
    ) -> Dict:
        """Run OD inference on a single PIL image."""
        task = self.config.task_prompt
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.size,
        )

        objects = self._parse_objects(parsed, allowed_classes)
        return {
            "model": "florence2",
            "objects": objects,
            "processing_time_ms": elapsed_ms,
            "raw_response": parsed,
        }

    def predict_path(
        self,
        image_path: str | Path,
        allowed_classes: Optional[Set[str]] = None,
    ) -> Dict:
        image_path = Path(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        result = self.predict_image(image, allowed_classes=allowed_classes)
        result["document_id"] = image_path.stem
        result["image_path"] = str(image_path)
        return result

    @staticmethod
    def _parse_objects(parsed: Dict, allowed_classes: Optional[Set[str]]) -> List[Dict]:
        task_key = "<OD>"
        task_result = parsed.get(task_key, parsed)
        if not isinstance(task_result, dict):
            return []

        labels = task_result.get("labels", []) or []
        bboxes = task_result.get("bboxes", []) or []
        objects = []
        for label, bbox in zip(labels, bboxes):
            label_str = str(label)
            if allowed_classes and label_str not in allowed_classes:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            objects.append(
                {
                    "label": label_str,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 1.0,
                }
            )
        return objects
