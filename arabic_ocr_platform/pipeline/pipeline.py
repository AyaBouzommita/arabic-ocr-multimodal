import time
import logging
import cv2

from arabic_ocr_platform.pipeline.nlp.config import NLPConfig
from arabic_ocr_platform.pipeline.nlp.corrector import NLPCorrector
from arabic_ocr_platform.pipeline.fusion.engine import FusionEngine

logger = logging.getLogger(__name__)

class OCRPipeline:
    """
    Main Orchestration Pipeline wiring together OCR, Vision, and NLP correction.
    """
    def __init__(self, vision_model=None, ocr_model=None, nlp_alpha=0.5):
        self.vision_model = vision_model
        self.ocr_model = ocr_model
        
        # Initialize sub-components
        logger.info("Initializing Pipeline components...")
        self.nlp_config = NLPConfig()
        self.nlp_corrector = NLPCorrector(config=self.nlp_config)
        self.fusion_engine = FusionEngine(alpha=nlp_alpha)

    def process(self, image_path: str, document_id: str) -> dict:
        """
        Process an image through the full pipeline.
        """
        logger.info(f"Processing document: {document_id}")
        
        # 1. Vision Stage
        start_time = time.time()
        image, detected_objects = self._run_vision_stage(image_path)
        logger.info(f"Vision stage completed in {time.time() - start_time:.2f}s")

        # 2. OCR Stage
        start_time = time.time()
        ocr_data = self._run_ocr_stage(image, detected_objects, document_id)
        logger.info(f"OCR stage completed in {time.time() - start_time:.2f}s")

        # 3. NLP Correction Stage
        start_time = time.time()
        correction_candidates = self.nlp_corrector.process(ocr_data)
        logger.info(f"NLP Correction stage completed in {time.time() - start_time:.2f}s")

        # 4. Fusion Stage
        start_time = time.time()
        final_output = self.fusion_engine.process(
            ocr_data=ocr_data,
            detected_objects=detected_objects,
            correction_candidates=correction_candidates
        )
        logger.info(f"Fusion stage completed in {time.time() - start_time:.2f}s")

        return final_output

    def _run_vision_stage(self, image_path: str):
        """Run YOLOv11s to detect text regions and crop them."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        h, w = image.shape[:2]
        detected_objects = []
        
        if self.vision_model is None:
            return image, detected_objects

        results = self.vision_model.predict(
            source=str(image_path),
            conf=0.25,
            imgsz=640,
            verbose=False,
        )

        class_names = self.vision_model.names
        TEXT_CLASSES = {"text", "table", "letters"}
        ARABIC_LETTER_CLASSES = {
            "aleph", "baaa", "daal", "dad", "faaa", "geem", "haaa", "hamza",
            "hamzasater", "kaaf", "lam", "mem", "non", "qaf", "raaa", "sad",
            "sen", "sheen", "taaa", "thaa", "thal", "then", "ttaa", "waaa",
            "yaaa", "zaaa", "3en", "5aaa", "5en",
        }

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cls_name = class_names.get(cls_id, "unknown")

                    if cls_name in TEXT_CLASSES or cls_name in ARABIC_LETTER_CLASSES:
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        
                        pad = 5
                        x1 = max(0, int(x1) - pad)
                        y1 = max(0, int(y1) - pad)
                        x2 = min(w, int(x2) + pad)
                        y2 = min(h, int(y2) + pad)
                        
                        if (x2 - x1) < 10 or (y2 - y1) < 10:
                            continue

                        detected_objects.append({
                            "label": cls_name,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "crop": image[y1:y2, x1:x2].copy()
                        })

        # Sort regions: top-to-bottom, then left-to-right
        detected_objects.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return image, detected_objects

    def _run_ocr_stage(self, image, detected_objects: list, document_id: str) -> dict:
        """Run PaddleOCR on crops (or full image) and format to schema."""
        start_time = time.perf_counter()
        
        all_tokens = []
        all_text_parts = []
        
        if self.ocr_model is None:
            return {"document_id": document_id, "engine": "paddleocr", "raw_text": "", "tokens": [], "processing_time_ms": 0}

        ocr = self.ocr_model._get_ocr()

        if not detected_objects:
            # Fallback to full image
            paddle_results = ocr.ocr(image, cls=self.ocr_model.use_angle_cls)
            if paddle_results and paddle_results[0]:
                for line in paddle_results[0]:
                    box_points = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    if not text.strip():
                        continue
                        
                    xs = [p[0] for p in box_points]
                    ys = [p[1] for p in box_points]
                    global_bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                    
                    all_tokens.append({"text": text.strip(), "bbox": global_bbox, "confidence": float(conf)})
                    all_text_parts.append(text.strip())
        else:
            for region in detected_objects:
                crop = region["crop"]
                bbox_offset = region["bbox"]
                
                paddle_results = ocr.ocr(crop, cls=self.ocr_model.use_angle_cls)
                if paddle_results and paddle_results[0]:
                    for line in paddle_results[0]:
                        box_points = line[0]
                        text = line[1][0]
                        conf = line[1][1]
                        
                        if not text.strip():
                            continue
                            
                        xs = [p[0] for p in box_points]
                        ys = [p[1] for p in box_points]
                        global_bbox = [
                            float(min(xs)) + bbox_offset[0],
                            float(min(ys)) + bbox_offset[1],
                            float(max(xs)) + bbox_offset[0],
                            float(max(ys)) + bbox_offset[1],
                        ]
                        
                        all_tokens.append({"text": text.strip(), "bbox": global_bbox, "confidence": float(conf)})
                        all_text_parts.append(text.strip())

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        # fix direction for display or fusion
        def fix_arabic_direction(text: str) -> str:
            if not text:
                return text
            words = text.split()
            fixed_words = []
            for word in words:
                if any('\u0600' <= char <= '\u06FF' for char in word):
                    if word.startswith(('ة', 'ى', 'ا', 'و', 'ر', 'ز', 'د', 'ذ')):
                        fixed_words.append(word[::-1])
                    else:
                        fixed_words.append(word)
                else:
                    fixed_words.append(word)
            return " ".join(fixed_words)

        raw_text = fix_arabic_direction(" ".join(all_text_parts))
        
        return {
            "document_id": document_id,
            "engine": "yolo+paddleocr",
            "raw_text": raw_text,
            "tokens": all_tokens,
            "processing_time_ms": elapsed_ms
        }
