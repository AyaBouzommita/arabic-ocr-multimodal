# Interface Contracts — Shared Data Schemas

> **Version**: v1.0 — Sprint 0 (June 2026)
> **Status**: FROZEN — any change must be agreed by the whole team

These schemas define the exact JSON structures produced and consumed by every
module in the pipeline. All OCR engines, detection models, and NLP modules
**must** validate against these schemas from Sprint 1 onward.

---

## 1. OCR Output Schema

**Produced by**: Tesseract, EasyOCR, PaddleOCR  
**File**: `ocr_output.schema.json`

```json
{
  "document_id": "string",
  "engine": "tesseract | easyocr | paddleocr",
  "raw_text": "string",
  "tokens": [
    {
      "text": "string",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }
  ],
  "processing_time_ms": 0
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | Unique document identifier (filename stem) |
| `engine` | string | OCR engine name: `tesseract`, `easyocr`, or `paddleocr` |
| `raw_text` | string | Full extracted text concatenated |
| `tokens` | array | Per-word tokens with spatial and confidence info |
| `tokens[].text` | string | Recognized word text |
| `tokens[].bbox` | array[4] | Bounding box `[x1, y1, x2, y2]` in pixels |
| `tokens[].confidence` | float | Engine confidence score (0.0–100.0) |
| `processing_time_ms` | integer | Total processing time in milliseconds |

---

## 2. Detection Output Schema

**Produced by**: YOLOv8, Detectron2, Florence-2/PP-YOLOE  
**File**: `detection_output.schema.json`

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

**Consumed by**: The correction/fusion stage  
**File**: `fusion_input.schema.json`

```json
{
  "document_id": "string",
  "ocr": { "...": "see ocr_output.schema.json" },
  "detected_objects": [ "...": "see detection_output.schema.json" ],
  "correction_candidates": [
    {
      "original_word": "string",
      "position": [start, end],
      "candidates": [
        { "text": "string", "lm_score": 0.0 }
      ]
    }
  ]
}
```

---

## Validation

- Every module producing OCR or detection output must include a unit test
  validating its output against these schemas.
- Schema validation is enforced from Sprint 1 onward via CI (`pytest`).
- See `tests/test_interface_contracts.py` for the validation test suite.
