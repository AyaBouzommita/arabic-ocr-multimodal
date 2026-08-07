# Neoledge Arabic-English OCR Pipeline

A highly robust multimodal OCR pipeline designed for mixed Arabic and English documents. This project was built during the 2026 Summer Internship at NeoLedge and combines spatial Computer Vision (YOLO) with Natural Language Processing (AraBERT & RoBERTa) to achieve state-of-the-art text extraction accuracy.

## Architecture

The pipeline processes images through 4 distinct stages:

1. **Bounding Box Detection (YOLOv11s)**: Extracts clean document sections to eliminate layout noise.
2. **Text Extraction (EasyOCR)**: Specially tuned parameters for maximum sensitivity on punctuation and symbols.
3. **Numeral Normalization**: Automatically converts Eastern Arabic Numerals (`٠-٩`) to Western digits (`0-9`).
4. **Contextual Correction (AraBERT & RoBERTa)**: Masked Language Models (MLM) fortified by edit-distance gating and `pyspellchecker` dictionaries ensure high-fidelity spelling correction without hallucination. Features sliding window context for documents containing 1000+ words.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run testing pipeline on your images:
   ```bash
   python scripts/test_single_image.py
   ```

## Detailed Walkthrough

For a comprehensive explanation of how the pipeline was constructed and the benchmark performance metrics, please see [WALKTHROUGH.md](./WALKTHROUGH.md).
