# Arabic Document OCR — Layout Detection Model Selection

## Goal

This sprint focuses on training and comparing multiple YOLO versions for **document layout detection** on Arabic documents. The selected model will be integrated with **PaddleOCR** (the best OCR engine identified in Sprint 1) to build a full pipeline: detect regions → extract text.

## Dataset

We merged **5 different datasets** into a single unified dataset:

| Source Dataset | Content | Original Classes |
|---|---|---|
| `ar-data` | Individual Arabic letters | 30 classes (aleph, baaa, lam, etc.) |
| `en-data` | English business cards | email, phone, company, etc. |
| `ocr_arabicv1` | Arabic words | arabic-words |
| `ocr_arabicv5` | Arabic documents | text, picture, qr_code, signature |
| `ocr_mixv2` | English invoices | table, invoice_no, total_amount, etc. |

### Unified Taxonomy (36 classes)

All classes were mapped into two categories:

- **General Layout**: `text`, `table`, `picture`, `signature`, `stamp`, `qr_code`
- **Arabic Alphabet**: `aleph`, `baaa`, `daal`, `dad`, `faaa`, `geem`, `haaa`, `hamza`, `hamzasater`, `kaaf`, `lam`, `mem`, `non`, `qaf`, `raaa`, `sad`, `sen`, `sheen`, `taaa`, `thaa`, `thal`, `then`, `ttaa`, `waaa`, `yaaa`, `zaaa`, `3en`, `5aaa`, `5en`, `letters`

### Structure

```
data/dataset_for_object_detection/
├── dataset.yaml
├── train/
│   ├── images/
│   ├── labels/
│   └── annotations/
├── valid/
│   ├── images/
│   ├── labels/
│   └── annotations/
└── test/
    ├── images/
    ├── labels/
    └── annotations/
```

## Trained Models

Each model has its own training script in `scripts/`:

| Script | Model | Size | Epochs |
|---|---|---|---|
| `train_yolov8s.py` | YOLOv8 Small | 11M params | 50 |
| `train_yolov8m.py` | YOLOv8 Medium | 25M params | 50 |
| `train_yolov10s.py` | YOLOv10 Small | 8M params | ❌ Skipped (too slow) |
| `train_yolov11s.py` | YOLOv11 Small | 9M params | 50 |
| `train_yolov11m.py` | YOLOv11 Medium | 20M params | 37 (early stopping) |

**Hardware**: NVIDIA GeForce RTX 3050 Ti (4 GB VRAM), PyTorch 2.5.1+cu118

## Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8n | 0.465 | 0.703 | 0.666 | 0.399 |
| YOLOv8s | 0.706 | 0.762 | 0.788 | 0.414 |
| YOLOv8m | 0.708 | 0.752 | 0.844 | 0.458 |
| **YOLOv11s** | **0.715** | **0.745** | **0.821** | **0.465** |
| YOLOv11m | 0.663 | 0.825 | 0.846 | 0.462 |

Trained weights and training curves are stored in `runs/detect/evaluation/yolo_comparison/`.

## Why YOLOv11s?

### YOLOv11s vs YOLOv11m

YOLOv11m has a slightly higher mAP50 (0.846 vs 0.821), but:

- **Lower precision** (0.663 vs 0.715) → more false detections, which produces garbage OCR text downstream
- **2x heavier model** → slower inference and higher VRAM usage
- **Batch size reduced to 8** due to the 4 GB VRAM limit, slowing down training

### YOLOv11s vs YOLOv8m

YOLOv8m has a higher mAP50 (0.844 vs 0.821), but:

- **Lower mAP50-95** (0.458 vs 0.465) → YOLOv11s produces more accurate bounding boxes
- **Lower precision** (0.708 vs 0.715)
- **Much heavier model** (25M vs 9M params) → 3x slower at inference
- **Older architecture** (v8 vs v11)

### YOLOv11s vs YOLOv8s

- **Better on all metrics** except recall (0.745 vs 0.762)
- **mAP50**: 0.821 vs 0.788
- **mAP50-95**: 0.465 vs 0.414
- **Newer architecture** with better optimizations

### Conclusion

**YOLOv11s** offers the best trade-off between:

- ✅ **Highest precision** (0.715) → fewest false positives
- ✅ **Highest mAP50-95** (0.465) → most accurate bounding boxes
- ✅ **Lightweight model** → fast inference, fits on RTX 3050 Ti (4 GB)
- ✅ **Latest architecture** (YOLO 11)

## How to Run

```bash
# Activate environment
.\venv\Scripts\activate

# Train a model
python scripts/train_yolov11s.py

# Results will be saved to runs/detect/evaluation/yolo_comparison/yolov11s/
```

## Next Steps

1. Integrate YOLOv11s with PaddleOCR into a full pipeline
2. Test the pipeline on real Arabic documents
3. Evaluate OCR quality with and without layout detection
