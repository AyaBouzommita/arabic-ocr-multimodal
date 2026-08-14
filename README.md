# Neoledge OCR: Bilingual Multimodal Pipeline

This repository contains the complete codebase for the Neoledge OCR project, developed during the summer 2026 internship. It features a robust multi-layered architectural approach to text extraction, explicitly designed to eliminate hallucinations when processing complex Arabic, French, and English bilingual documents.

## 🏗️ Architecture Layers
1. **YOLOv11s Layout Parser:** Crops paragraphs and tables to prevent OCR layout confusion.
2. **Dual OCR Pipelines:** Uses EasyOCR for Arabic/English and PaddleOCR for French/English.
3. **AraBERT / RoBERTa:** Masked Language Modeling engines to contextually correct spelling.
4. **Strict Dictionary Validation:** Ensures all post-processed words actually exist.
5. **Gemini Reconstructor:** Injects clean Markdown formatting into the raw string outputs.
6. **Colab VLM API:** Offloads ultra-dense documents to a heavy Qwen2-VL model running on a T4 GPU.

---

## 🚀 Getting Started

### 1. Installation

First, clone this repository to your local machine:
```bash
git clone https://github.com/AyaBouzommita/arabic-ocr-multimodal.git
cd arabic-ocr-multimodal
```

Install the required Python dependencies (it is highly recommended to use a virtual environment):
```bash
pip install -r requirements.txt
```
*(Note: Ensure you have PyTorch installed with CUDA support if you intend to run YOLOv11s and AraBERT on your local GPU).*

### 2. Environment Variables

To utilize the formatting reconstructor and the heavy Colab VLM, you need to set up the following environment variables:

- `GEMINI_API_KEY` (Required for Markdown formatting)
- `COLAB_API_URL` (Required only if you are running the Ngrok/Colab proxy bridge)

**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
$env:COLAB_API_URL="https://your-ngrok-url.ngrok.app"
```

**On Mac/Linux:**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
export COLAB_API_URL="https://your-ngrok-url.ngrok.app"
```

### 3. Running the Django Server

Navigate into the Django project directory and start the server:

```bash
cd neoledge_web
python manage.py runserver
```

The web platform will now be accessible at `http://127.0.0.1:8000/`. From here, you can access the frontend UI, upload documents, and download the Python API client for integration into other applications.

### 4. Downloading Models
The first time you run the pipeline, the system will automatically download the required HuggingFace models (AraBERT, RoBERTa) and YOLOv11s weights. Please ensure you have a stable internet connection and sufficient disk space.

---

## 🛠️ API Usage
If you download the `neoledge_api_client.py` script from the web platform, you can seamlessly integrate this pipeline into your own automated workflows.

```python
from neoledge_api_client import NeoledgeOCR

# Initialize with the local server URL
client = NeoledgeOCR("http://127.0.0.1:8000")

# Extract Markdown from a local document
markdown_result = client.extract("path/to/document.jpg")
print(markdown_result)
```

## ❤️ Credits
Developed during the 2026 Neoledge Summer Internship.
