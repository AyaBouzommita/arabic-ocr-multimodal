# Visual Annotation Schema

This document defines the classes, bounding box formats, and annotation protocols used for visual document element detection in Sprint 2.

---

## 1. Class Definitions

We annotate 7 target visual elements:

| Class ID | Class Name | Description | Example Visual Cues |
|---|---|---|---|
| `0` | `Stamp` | Official cachets, stamps, ink marks (round, oval, rectangular) | Blue/red ink circles, official state seal |
| `1` | `Logo` | Institutional and corporate brandings | Logo of NeoLedge, Ministry logo |
| `2` | `Signature` | Handwritten signatures or initials | Pen strokes, scribbles at document bottom |
| `3` | `Header` | Document top headers containing institutional metadata | Text lines with contact info, document titles |
| `4` | `Table` | Tabular visual structures | Rows and columns, gridlines |
| `5` | `Date` | Date stamps or written dates | "Zarzis, le 15/06/2026" or "التاريخ: ٢٠٢٦/٠٦/١٥" |
| `6` | `InstitutionName` | Text representing the organization name | "NeoLedge", "وزارة العدل" |

---

## 2. Bounding Box Formats

### YOLO Annotation Format
For training YOLOv8, annotations must be exported in **YOLO format**:
* One `.txt` file per image.
* Each line contains: `<class_idx> <x_center> <y_center> <width> <height>`
* Coordinates must be normalized between `0.0` and `1.0` relative to image dimensions.

### Pascal VOC / COCO Formats
For Detectron2, the dataset will be loaded in COCO JSON format:
* Bounding Box: `[x_min, y_min, width, height]` (pixel absolute values).

---

## 3. Data Split Strategy
To ensure reproducible evaluations, the dataset is split as follows:
* **Train**: 70% (~105 images)
* **Val**: 15% (~22 images)
* **Test**: 15% (~23 images)

Annotation work is split equally among interns (50 images per intern) in CVAT/LabelImg, and merged into a single repository before training.
