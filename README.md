# Arabic OCR Multimodal Platform

Amélioration de la reconnaissance de caractères arabes par fusion entre OCR, compréhension contextuelle et analyse visuelle du document.

Reference: `#NEO-STAGE-ETE-2026-05`  
Company: **NeoLedge — Zarzis Smart Center, Tunisia**

---

## 1. Project Overview

This repository implements a multimodal Arabic OCR pipeline that fuses:
1. **Raw OCR Text**: Text extracted by Tesseract, EasyOCR, or PaddleOCR.
2. **Linguistic Context**: Arabic Language Models (AraBERT / CAMeL-BERT / AraGPT2) for sentence-level candidate scoring.
3. **Visual Document Analysis**: Object Detection (YOLOv8 / Detectron2 / Florence-2) identifying visual elements (stamps, signatures, logos, tables) to guide corrections.

---

## 2. Directory Structure

```
arabic-ocr-multimodal/
├── .github/
│   ├── workflows/
│   │   └── ci.yml             # CI/CD pipeline
│   └── pull_request_template.md
├── arabic_ocr_platform/       # Django Platform Root
│   ├── apps/                  # Django Apps
│   │   ├── documents/         # Document uploads and storage
│   │   ├── evaluation/        # CER/WER runs tracking
│   │   └── hitl/              # Human-in-the-loop validation
│   ├── config/                # Django Configurations
│   │   ├── settings/          # Split settings
│   │   │   ├── base.py
│   │   │   ├── development.py
│   │   │   └── production.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── pipeline/              # Independent AI Pipeline Modules
│   │   ├── evaluation/        # CER/WER scoring & Arabic normalizer
│   │   ├── fusion/            # Multimodal fusion engine
│   │   ├── nlp/               # Language models correction & ranking
│   │   ├── ocr/               # OCR engine connectors
│   │   ├── utils/             # JSON Schema Validators
│   │   └── vision/            # YOLO detectors wrappers
│   └── manage.py
├── data/
│   └── corpus/                # Raw corpus documents (.gitkeep)
├── docs/
│   ├── annotation_schema.md   # Visual annotations guidelines
│   ├── git_workflow.md        # Branching and Commit guidelines
│   └── interface_contracts.md # Shared JSON Schemas documentation
├── reports/                   # Performance & benchmark reports (.gitkeep)
├── results/                   # Evaluator execution metrics (.gitkeep)
├── schemas/                   # Shared JSON Schemas (Frozen contracts)
│   ├── ocr_output.schema.json
│   ├── detection_output.schema.json
│   └── fusion_input.schema.json
├── .env.example               # Template environment configuration
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+
* Git

### Local Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd arabic-ocr-multimodal
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure local environment variables:
   ```bash
   cp .env.example .env
   ```

5. Run Django system check to verify setup:
   ```bash
   python arabic_ocr_platform/manage.py check
   ```

---

## 4. Running the CER/WER Evaluator & Tests

### Running Tests
To run all test suites (including Arabic normalization, CER/WER distance computation, and JSON Schema validators):
```bash
pytest
```

### Running Scorer manually
You can import `calculate_cer` and `calculate_wer` directly in your scripts:
```python
from pipeline.evaluation.evaluator import calculate_cer, calculate_wer

ref = "كَتَبَ أحمدُ الدرسَ."
hyp = "كتب احمد الدرس"

print("CER:", calculate_cer(ref, hyp, normalize=True)) # Outputs: 0.0
print("WER:", calculate_wer(ref, hyp, normalize=True)) # Outputs: 0.0
```

---

## 5. Collaboration & Git Guidelines

Please refer to the following documentation files under `docs/`:
* **Git Branches and Commits**: [docs/git_workflow.md](docs/git_workflow.md)
* **JSON Schema Contracts**: [docs/interface_contracts.md](docs/interface_contracts.md)
* **Dataset Annotation Schema**: [docs/annotation_schema.md](docs/annotation_schema.md)
