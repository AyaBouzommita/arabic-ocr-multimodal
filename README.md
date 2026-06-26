To test and see the results, you have three different ways to execute the project depending on what you want to see. 

### 1. Run the Baseline Evaluation on the Entire Corpus
To process all images in your `data/raw/` folder, calculate the CER/WER against the ground truth, and generate the final baseline CSV reports required for your sprint:

```powershell
.\venv\Scripts\python -m scripts.run_tesseract_baseline --save-json
```
* **What it does:** Runs Tesseract on every image, computes error rates, and saves the results.
* **Where to see results:** Check the terminal output for a summary table, and look at the generated files in `results/sprint1_baseline.csv` and the `results/json/` folder.

### 2. Run Tesseract on a Single Image (Quick Test)
If you want to quickly test a specific image and see the detailed JSON output (including the bounding boxes and confidence scores for every word):

```powershell
.\venv\Scripts\python main.py data/raw/sample.png
```
* **Add Evaluation:** You can add the `--evaluate` flag to also calculate the CER/WER for that specific image if it has a corresponding `.txt` ground truth file:
  ```powershell
  .\venv\Scripts\python main.py data/raw/sample.png --evaluate
  ```
* **Simple Output:** If you just want to see the raw extracted Arabic text without all the JSON metadata, use the `--simple` flag:
  ```powershell
  .\venv\Scripts\python main.py data/raw/sample.png --simple
  ```

### 3. Run the Automated Test Suite (QA)
To verify that everything in the codebase is functioning correctly and strictly follows the Sprint 0 interface contracts:

```powershell
.\venv\Scripts\python -m pytest tests/ -v
```
* **What it does:** Runs all 40 automated tests we wrote, verifying things like image preprocessing, schema validation, and JiWER score calculations. All of them should show as `PASSED`.
