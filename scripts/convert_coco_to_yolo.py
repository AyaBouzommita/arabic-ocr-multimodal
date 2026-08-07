import os
import json
from pathlib import Path
from tqdm import tqdm

UNIFIED_DIR = Path("data/unified_dataset")
SPLITS = ["train", "valid", "test"]

def convert_to_yolo():
    for split in SPLITS:
        print(f"Converting split: {split}")
        anno_path = UNIFIED_DIR / split / "annotations" / "_annotations.coco.json"
        out_labels_dir = UNIFIED_DIR / split / "labels"
        os.makedirs(out_labels_dir, exist_ok=True)
        
        if not anno_path.exists():
            print(f"Warning: {anno_path} not found!")
            continue
            
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Create image lookup
        img_id_to_info = {img["id"]: img for img in data.get("images", [])}
        
        # Group annotations by image_id
        img_id_to_anns = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)
            
        # Write YOLO txt files
        for img_id, img_info in tqdm(img_id_to_info.items(), desc=f"Writing YOLO labels for {split}"):
            img_width = img_info["width"]
            img_height = img_info["height"]
            img_filename = img_info["file_name"]
            
            # Change extension to .txt
            txt_filename = os.path.splitext(img_filename)[0] + ".txt"
            txt_path = out_labels_dir / txt_filename
            
            anns = img_id_to_anns.get(img_id, [])
            
            with open(txt_path, "w", encoding="utf-8") as out_f:
                for ann in anns:
                    cat_id = ann["category_id"]
                    x_min, y_min, w, h = ann["bbox"]
                    
                    # Calculate normalized center x, center y, width, height
                    x_center = (x_min + w / 2) / img_width
                    y_center = (y_min + h / 2) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height
                    
                    # Ensure within [0, 1] bounds (sometimes annotations are slightly out of bounds)
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))
                    
                    out_f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
if __name__ == "__main__":
    convert_to_yolo()
