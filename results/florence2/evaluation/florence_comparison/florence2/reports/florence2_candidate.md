# Florence-2 Candidate Report (Vision Bake-off 3/3)

## Candidate
- Model: Florence-2 (LoRA fine-tuned)
- Dataset: shared team YOLO dataset (`data/vision/detection/`)
- Branch: `florence2-detection-engine`

## Training Summary
- Train samples: N/A
- Validation samples: N/A
- Epochs: N/A
- Device: N/A
- Final train loss: N/A
- Final val loss: N/A

## Evaluation (test split)
- mAP@0.5: **0.4444**
- mAP@0.5:0.95 (class-avg): **0.4444**
- Avg inference speed: **350.4 ms/image**
- Evaluated images: 20

## Integration Effort
- Score (1=easy, 5=hard): **3/5**
- Notes: Requires transformers + peft + GPU for practical training; strong multi-task flexibility.

## Strengths
- One model can support detection and future multimodal tasks (caption/OCR/grounding).
- Works on the shared annotated dataset without reformatting the whole pipeline.

## Limitations
- Lower mAP than dedicated detectors is common for Florence-2.
- Training and inference are slower without GPU.
- Class vocabulary is open-ended, so post-filtering by allowed classes is required.

## Artifacts
- Model: `results/florence2/model`
- Metrics: `/content/arabic-ocr-multimodal/results/florence2/metrics.json`
- Predictions: `/content/arabic-ocr-multimodal/results/florence2/predictions`
