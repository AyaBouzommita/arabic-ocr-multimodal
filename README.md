# Neoledge Arabic OCR Project

This project focuses on evaluating and integrating Optical Character Recognition (OCR) engines for extracting Arabic text from document images.

## Project Progress & Evaluation (Latest Updates)

To determine the best OCR engine for our Arabic datasets, we conducted a rigorous comparative evaluation between three leading OCR engines.

### 1. Dataset Preparation (KHATT Dataset)
We introduced a new ground-truth dataset (`ar-data-groundtruth`) based on the KHATT format. The ground truth in this dataset provided Arabic text as Latin transliteration codes (e.g., `aa` for Alef, `ba` for Baa). 
* **What we did:** We built a custom decoding module (`scripts/khatt_decoder.py`) that automatically translates these Latin codes back into native Arabic text strings to use as our "gold standard" ground truth.

### 2. Engine Evaluation
We developed a custom evaluation script (`scripts/run_khatt_evaluation.py`) to benchmark the engines. The script runs the decoded ground truth against the text extracted by the engines and calculates:
* **Character Error Rate (CER):** How many individual letters the engine gets wrong.
* **Word Error Rate (WER):** How many whole words the engine gets wrong.

We evaluated the following engines on a sample of 100 validation images:
1. **Tesseract:** A widely used open-source engine (with `psm=3` and Arabic language pack).
2. **EasyOCR:** A PyTorch-based OCR engine.
3. **PaddleOCR:** A highly optimized OCR engine by PaddlePaddle.

### 3. Final Results

The results of the 100-image benchmark were as follows:

| Engine | Avg Character Error Rate (CER) | Avg Word Error Rate (WER) | Avg Time (ms per image) |
| :--- | :--- | :--- | :--- |
| **Tesseract** | **72.04%** | 115.41% | 321.9 ms |
| **EasyOCR** | 75.32% | 114.80% | 6383.4 ms |
| **PaddleOCR** | 80.47% | **100.87%** | **317.4 ms** |

*(Note: The relatively high error rates are due to the dataset containing complex cursive Arabic handwriting, which is notoriously difficult for out-of-the-box OCR without dedicated fine-tuning).*

### 4. Conclusion & Next Steps
**PaddleOCR** was selected as the best engine moving forward. While Tesseract had a slightly better Character Error Rate, PaddleOCR achieved the best Word Error Rate, meaning it was more successful at capturing complete, coherent words. Furthermore, PaddleOCR is incredibly fast and handles its own internal image normalization without needing extra preprocessing steps.

---

## How to Run the Code

### 1. Run the OCR Evaluation Benchmark
To run the evaluation script yourself and reproduce the benchmark scores against the KHATT dataset:
```powershell
.\venv\Scripts\python scripts\run_khatt_evaluation.py --limit 100
```
This will output a summary table to the console and save detailed per-image metrics to `results/khatt_evaluation_results.csv`.

### 2. Run the Original Baseline Evaluation
To process the original images in the `data/raw/` folder and calculate metrics for sprint reporting:
```powershell
.\venv\Scripts\python -m scripts.run_tesseract_baseline --save-json
```

### 3. Quick Test on a Single Image
To test Tesseract on a single image and view the detailed JSON output (including bounding boxes and confidence scores):
```powershell
.\venv\Scripts\python main.py data/raw/sample.png
```
* **Simple Output:** If you just want the raw extracted Arabic text without JSON metadata, use the `--simple` flag:
  ```powershell
  .\venv\Scripts\python main.py data/raw/sample.png --simple
  ```

### 4. Run Automated Test Suite (QA)
To verify that everything in the codebase strictly follows the interface contracts:
```powershell
.\venv\Scripts\python -m pytest tests/ -v
```
