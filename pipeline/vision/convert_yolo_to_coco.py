"""
Conversion YOLO -> COCO JSON pour Detectron2
Samar Zaabouti - Sprint 2
"""

import os
import json
import glob
from PIL import Image

DATASET_DIR = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_for_object_detection/dataset_for_object_detection"
OUTPUT_DIR  = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_coco"

# Classes du dataset.yaml
CLASSES = [
    "text", "table", "picture", "signature", "stamp",
    "qr_code", "3en", "5aaa", "5en", "aleph", "baaa",
    "daal", "dad", "faaa", "geem", "haaa", "hamza",
    "hamzasater", "kaaf", "lam", "mem", "non", "qaf",
    "raaa", "sad", "sen", "sheen", "taaa", "thaa",
    "thal", "then", "ttaa", "waaa", "yaaa", "zaaa", "letters"
]

def yolo_to_coco(split):
    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    }

    image_id = 0
    ann_id = 0

    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(images_dir, "*.png"))

    for img_path in sorted(image_files):
        try:
            img = Image.open(img_path)
            w, h = img.size
        except:
            continue

        filename = os.path.basename(img_path)
        coco["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        label_path = os.path.join(
            labels_dir,
            os.path.splitext(filename)[0] + ".txt"
        )

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, bw, bh = map(float, parts)
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    bw_px = bw * w
                    bh_px = bh * h

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(cls_id),
                        "bbox": [x1, y1, bw_px, bh_px],
                        "area": bw_px * bh_px,
                        "iscrowd": 0
                    })
                    ann_id += 1

        image_id += 1

    return coco


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Conversion train...")
train_coco = yolo_to_coco("train")
with open(f"{OUTPUT_DIR}/train.json", 'w') as f:
    json.dump(train_coco, f)
print(f"Train : {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")

print("Conversion valid...")
val_coco = yolo_to_coco("valid")
with open(f"{OUTPUT_DIR}/val.json", 'w') as f:
    json.dump(val_coco, f)
print(f"Val   : {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations")

print(f"\n✅ Conversion terminée ! Fichiers dans : {OUTPUT_DIR}")
print(f"Classes : {CLASSES[:6]}...")
