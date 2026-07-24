# Vision Detection Bake-off — 3-Way Comparison

Comparison used for the team decision between detection candidates.

| Candidate | mAP@0.5 | mAP@0.5:0.95 | Speed (ms/img) | Integration effort | Notes |
|---|---:|---:|---:|---:|---|
| YOLOv8 / PP-YOLOE | TBD | TBD | TBD | 5/5 | Pending teammate results |
| Detectron2 | TBD | TBD | TBD | 2/5 | Pending teammate results |
| Florence-2 (ours) | 0.4444 | 0.4444 | 350.4 | 3/5 | Requires transformers + peft + GPU for practical training; strong multi-task flexibility. |

## Decision Criteria
1. **Detection quality**: prioritize higher mAP on the shared test split.
2. **Runtime**: lower ms/image is better for production OCR pipeline latency.
3. **Integration effort**: simpler deployment and maintenance wins when scores are close.

## Current Recommendation
- Use Florence-2 if multimodal flexibility is strategic for later fusion stages.
- Use YOLO/PP-YOLOE if pure detection speed and mAP dominate.
- Use Detectron2 if the team already standardized on COCO/MMDetection tooling.

## Florence-2 Evidence
- Test images evaluated: 20
- Metrics file: `results/florence2/metrics.json`
