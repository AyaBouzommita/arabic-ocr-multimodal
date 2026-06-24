# Interface Contracts (Shared Data Schemas)

This document freezes the shared JSON schemas for OCR, Vision, and Fusion components. All candidate models and modules must produce and consume these structures.

---

## 1. OCR Output Schema
* **Produced by**: OCR candidates (Tesseract, EasyOCR, PaddleOCR)
* **File Path**: `/schemas/ocr_output.schema.json`

```json
{
  "document_id": "string",
  "engine": "tesseract | easyocr | paddleocr",
  "raw_text": "string",
  "tokens": [
    {"text": "string", "bbox": [x1, y1, x2, y2], "confidence": 0.0}
  ],
  "processing_time_ms": 0
}
```

---

## 2. Detection JSON Schema
* **Produced by**: Vision candidates (YOLOv8, Detectron2, Florence-2)
* **File Path**: `/schemas/detection_output.schema.json`

```json
{
  "document_id": "string",
  "model": "yolov8 | detectron2 | florence2",
  "objects": [
    {
      "label": "Stamp | Logo | Signature | Header | Table | Date | InstitutionName",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0,
      "context_tag": "administrative | financial | none"
    }
  ]
}
```

---

## 3. Fusion Input Schema
* **Consumed by**: Correction / Fusion stage
* **File Path**: `/schemas/fusion_input.schema.json`

```json
{
  "document_id": "string",
  "ocr": { 
    "document_id": "string",
    "engine": "tesseract | easyocr | paddleocr",
    "raw_text": "string",
    "tokens": [
      {"text": "string", "bbox": [x1, y1, x2, y2], "confidence": 0.0}
    ],
    "processing_time_ms": 0
  },
  "detected_objects": [
    {
      "label": "Stamp | Logo | Signature | Header | Table | Date | InstitutionName",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0,
      "context_tag": "administrative | financial | none"
    }
  ],
  "correction_candidates": [
    {
      "original_word": "string",
      "position": [start, end],
      "candidates": [
        {"text": "string", "lm_score": 0.0}
      ]
    }
  ]
}
```

---

## 4. Automatic Validation

To ensure every model conforms to these schemas, we use python's `jsonschema` library.

Example utility signature:
```python
from pipeline.utils.validator import validate_ocr_output

try:
    validate_ocr_output(data)
    print("Valid OCR Schema")
except SchemaValidationError as e:
    print(f"Validation failed: {e}")
```
All pipeline unit tests automatically validate generated test outputs against these files.
