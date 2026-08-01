"""Fast preparation of 300-sample fine-tuning dataset for YOLOv11s."""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

SRC_BASE = Path("data/OCR_GS_Data/ara/abhath")
DEST_DIR = Path("data/finetune_dataset")

TRAIN_COUNT = 250
VAL_COUNT = 50
TOTAL_COUNT = TRAIN_COUNT + VAL_COUNT

def main():
    print("=" * 60)
    print("  Fast 300-sample Fine-Tuning Dataset Preparation")
    print("=" * 60)

    # Find candidate subfolders
    subfolders = [f for f in SRC_BASE.iterdir() if f.is_dir()]
    print(f"[Scanning] Found {len(subfolders)} subfolders under {SRC_BASE}...")

    valid_pairs = []
    for folder in subfolders:
        for gt_file in folder.glob("*.gt.txt"):
            text = gt_file.read_text(encoding="utf-8").strip()
            if len(text) < 3:
                continue

            base = gt_file.name.replace(".gt.txt", "")
            img_file = folder / f"{base}.png"
            if img_file.exists() and img_file.stat().st_size > 500:
                valid_pairs.append((img_file, gt_file, base))
                if len(valid_pairs) >= 500:
                    break
        if len(valid_pairs) >= 500:
            break

    print(f"   Collected {len(valid_pairs)} candidate pairs.")

    random.shuffle(valid_pairs)
    selected = valid_pairs[:TOTAL_COUNT]
    train_samples = selected[:TRAIN_COUNT]
    val_samples = selected[TRAIN_COUNT:]

    print(f"   Splitting into {len(train_samples)} train and {len(val_samples)} val samples.")

    # Reset directory
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)

    for split in ["train", "val"]:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_path, gt_path, base_name in samples:
            # Copy image
            shutil.copy2(img_path, DEST_DIR / "images" / split / img_path.name)
            # Write YOLO label (Class 0: text)
            label_file = DEST_DIR / "labels" / split / f"{img_path.stem}.txt"
            label_file.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    # Copy dataset.yaml into place
    yaml_content = f"""path: {DEST_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: text
  1: table
  2: picture
  3: signature
  4: stamp
  5: qr_code
  6: 3en
  7: 5aaa
  8: 5en
  9: aleph
  10: baaa
  11: daal
  12: dad
  13: faaa
  14: geem
  15: haaa
  16: hamza
  17: hamzasater
  18: kaaf
  19: lam
  20: mem
  21: non
  22: qaf
  23: raaa
  24: sad
  25: sen
  26: sheen
  27: taaa
  28: thaa
  29: thal
  30: then
  31: ttaa
  32: waaa
  33: yaaa
  34: zaaa
  35: letters
"""
    (DEST_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  Fine-Tuning Dataset ready at: {DEST_DIR.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
