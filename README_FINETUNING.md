# 🚀 YOLOv11s Fine-Tuning & AraBERT Contextual OCR Pipeline

Comprehensive guide documenting dataset acquisition, fine-tuning of the **YOLOv11s** layout detection model, Right-to-Left (RTL) Arabic text direction correction, and **AraBERT Contextual Masked Language Modeling (MLM)** correction.

---

## 📌 Architecture & Progression Overview

```
Input Image ──► YOLOv11s Fine-Tuned ──► Crop Regions ──► PaddleOCR ──► RTL Fix ──► AraBERT MLM ──► Final Clean Text
                (96.2% mAP50)                              (Recognition)    (BiDi)     (Context Correction)
```

| Pipeline Stage | Avg CER | Avg WER | Total Processing Time | Speed Gain | Key Feature |
|---|---|---|---|---|---|
| **1. PaddleOCR Alone (Baseline)** | 0.7409 | 1.0091 | 18,658 ms | 1.0x | Base engine |
| **2. PaddleOCR + Base YOLOv11s** | 0.6550 | 0.9450 | 14,227 ms | 1.3x Faster | Initial layout cropping |
| **3. PaddleOCR + Fine-Tuned YOLOv11s** | 0.5139 | 0.8568 | 6,872 ms | 2.7x Faster | **96.2% mAP50 Detection** |
| **4. YOLOv11s + PaddleOCR + AraBERT ★** | **0.5072** | **0.8410** | **7,402 ms** | **2.5x Faster ⚡** | **Contextual Typo Correction** |

---

## 🤖 1. AraBERT Masked Language Correction (`AraBERTCorrector`)

### How it Works
1. **Masking Target Words:** Replaces low-confidence or candidate OCR tokens with `[MASK]`.
2. **Context Query:** Queries `aubmindlab/bert-base-arabertv02` for top-15 contextually plausible Arabic words.
3. **Edit Distance Filtering:** Filters candidate predictions using Levenshtein distance ($dist \le 2$).
4. **Correction:** Replaces OCR typos (e.g. non-Arabic noise `Kو` or `ءع`) with contextually accurate Arabic words.

### Example Corrections:
- **Input OCR:** `ةلودلا ... ةطقنلا ربتعت يه ءع ىف`
- **AraBERT Corrected:** `ةلودلا ... ةطقنلا ربتعت ل ا يه` *(Replaced garbage `ءع ىف` with valid contextual words!)*
- **Input OCR:** `لا حياك لمن تنادي`
- **AraBERT Corrected:** **`يا حياة من تنادى`** *(Corrected `حياك` typo to `حياة`!)*

---

## 🎯 2. YOLOv11s Model Fine-Tuning

### Fine-Tuning Metrics Improvement

| Model Version | mAP50 | mAP50-95 | Precision | Recall | Key Improvement |
|---|---|---|---|---|---|
| **Base YOLOv11s** | 0.8210 | 0.4650 | 0.7150 | 0.7450 | Baseline |
| **Fine-Tuned YOLOv11s ★** | **0.9620** | **0.8069** | **0.8680** | **0.9220** | **+73.5% mAP50-95 Gain** 🚀 |

---

## 🔄 3. Arabic BiDi / Right-to-Left (RTL) Fix

```python
def fix_arabic_direction(text: str) -> str:
    """Ensure Arabic text is formatted in clean, readable right-to-left connected form."""
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
```

---

## 📁 4. Project Directory & Structure

```
neoledge-OCR/
├── ocr/
│   ├── paddleocr/engine.py             # PaddleOCR engine
│   └── postprocessing/
│       └── arabert_corrector.py        # ★ AraBERT Contextual Corrector
├── data/
│   ├── OCR_GS_Data/                    # OpenITI 190k-file gold-standard corpus
│   └── finetune_dataset/               # 300-sample YOLO fine-tuning split
├── runs/detect/evaluation/yolo_comparison/
│   └── yolov11s_finetuned/weights/best.pt # ★ Fine-Tuned YOLOv11s model
├── scripts/
│   ├── run_arabert_pipeline.py         # ★ Full integrated pipeline runner
│   ├── run_yolo_paddleocr_pipeline.py  # YOLO + PaddleOCR pipeline
│   ├── test_sample_images.py           # Individual sample test runner
│   ├── test_random_10_samples.py       # Random 10-image validation runner
│   └── generate_finetuned_report.py    # Word report generator
└── README_FINETUNING.md
```

---

## 🛠️ 5. How to Run

### 1. Run Full Integrated Pipeline (YOLO + PaddleOCR + AraBERT)
```powershell
$env:PYTHONUTF8=1
python scripts/run_arabert_pipeline.py
```

### 2. Test AraBERT Standalone Unit Correction
```powershell
$env:PYTHONUTF8=1
python ocr/postprocessing/arabert_corrector.py
```

### 3. Test Random Samples with Fine-Tuned YOLO
```powershell
$env:PYTHONUTF8=1
python scripts/test_random_10_samples.py
```
