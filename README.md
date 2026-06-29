To test and see the results, you have three different ways to execute the project depending on what you want to see.

### 1. Run the Baseline Evaluation on the Entire Corpus
To process all images in your `data/raw/` folder, calculate the CER/WER against the ground truth, and generate the final baseline CSV reports required for your sprint:

```powershell
.\.venv\Scripts\python -m scripts.run_paddleocr_baseline --save-json
```
* **What it does:** Runs PaddleOCR on every image, computes error rates, and saves the results.
* **Where to see results:** Check the terminal output for a summary table, and look at the generated files in `results/sprint1_baseline.csv` and the `results/json/` folder.

### 2. Run PaddleOCR on a Single Image (Quick Test)
If you want to quickly test a specific image and see the detailed JSON output (including the bounding boxes and confidence scores for every word):

```powershell
.\.venv\Scripts\python main.py data/raw/sample.png
```
* **Add Evaluation:** You can add the `--evaluate` flag to also calculate the CER/WER for that specific image if it has a corresponding `.txt` ground truth file:
  ```powershell
  .\.venv\Scripts\python main.py data/raw/sample.png --evaluate
  ```
* **Simple Output:** If you just want to see the raw extracted Arabic text without all the JSON metadata, use the `--simple` flag:
  ```powershell
  .\.venv\Scripts\python main.py data/raw/sample.png --simple
  ```

### 3. Run the Automated Test Suite (QA)
To verify that everything in the codebase is functioning correctly and strictly follows the Sprint 0 interface contracts:

```powershell
.\.venv\Scripts\python -m pytest tests/ -v
```
* **What it does:** Runs all automated tests, verifying interface contract validation, CER/WER scoring, and image preprocessing. Tests requiring the PaddleOCR model and a sample image are automatically skipped if the image is not present.

### 4. PaddleOCR-specific Options
```powershell
# Disable angle classification (faster but less robust for rotated documents)
.\.venv\Scripts\python -m scripts.run_paddleocr_baseline --no-angle-cls

# Enable GPU inference (requires CUDA + GPU-enabled PaddlePaddle)
.\.venv\Scripts\python -m scripts.run_paddleocr_baseline --gpu

# Enable Otsu binarisation preprocessing
.\.venv\Scripts\python -m scripts.run_paddleocr_baseline --threshold
```

### 5. Install Dependencies
```powershell
pip install paddlepaddle paddleocr opencv-python Pillow jiwer pandas pytest
```
> **Note:** On first run, PaddleOCR automatically downloads the Arabic detection and recognition model weights (~100 MB). Make sure you have an internet connection.
