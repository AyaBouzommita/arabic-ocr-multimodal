import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

DATASETS_DIR = Path("data/data_training")
UNIFIED_DIR = Path("data/unified_dataset")

SPLITS = ["train", "valid", "test"]

# Define the unified taxonomy
UNIFIED_CATEGORIES = [
    "text", "table", "picture", "signature", "stamp", "qr_code",
    "3en", "5aaa", "5en", "aleph", "baaa", "daal", "dad", "faaa", "geem", 
    "haaa", "hamza", "hamzasater", "kaaf", "lam", "mem", "non", "qaf", 
    "raaa", "sad", "sen", "sheen", "taaa", "thaa", "thal", "then", 
    "ttaa", "waaa", "yaaa", "zaaa", "letters"
]

# Create a mapping from category name to Unified Category ID
CAT_NAME_TO_ID = {name: i for i, name in enumerate(UNIFIED_CATEGORIES)}

# Mapping rules for old class names to new unified class names
CLASS_MAPPING = {
    # Text types
    "arabic-words": "text",
    "texts": "text",
    "address": "text",
    "company": "text",
    "email": "text",
    "name": "text",
    "phone": "text",
    "social_media": "text",
    "title": "text",
    "website": "text",
    "information of the invoice": "text",
    "OMRON_address": "text",
    "OMRON_name": "text",
    "bill_to": "text",
    "currency": "text",
    "date": "text",
    "invoice_no": "text",
    "ship_to": "text",
    "total_amount": "text",
    "-": "text",
    "text": "text",

    # Table
    "table": "table",

    # Picture
    "picture": "picture",

    # Signature
    "signature": "signature",

    # Stamp
    "company_chop": "stamp",
    "stamp": "stamp",

    # QR Code
    "qr_code": "qr_code"
}

def get_unified_category_id(old_cat_name):
    mapped_name = CLASS_MAPPING.get(old_cat_name, old_cat_name)
    if mapped_name in CAT_NAME_TO_ID:
        return CAT_NAME_TO_ID[mapped_name]
    print(f"Warning: Category '{old_cat_name}' not mapped! Defaulting to 'text'")
    return CAT_NAME_TO_ID["text"]

def init_unified_coco():
    return {
        "info": {"description": "Unified Arabic Document Layout Dataset"},
        "licenses": [],
        "categories": [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(UNIFIED_CATEGORIES)],
        "images": [],
        "annotations": []
    }

def main():
    # Setup directories
    for split in SPLITS:
        os.makedirs(UNIFIED_DIR / split / "images", exist_ok=True)
        os.makedirs(UNIFIED_DIR / split / "annotations", exist_ok=True)

    for split in SPLITS:
        print(f"Processing split: {split}")
        unified_coco = init_unified_coco()
        image_id_counter = 0
        annotation_id_counter = 0

        # Iterate over all dataset folders in data_training
        for ds_name in os.listdir(DATASETS_DIR):
            ds_path = DATASETS_DIR / ds_name
            if not ds_path.is_dir():
                continue

            split_path = ds_path / split
            anno_path = split_path / "_annotations.coco.json"

            if not anno_path.exists():
                print(f"Skipping {ds_name}/{split} (no _annotations.coco.json)")
                continue

            print(f"  Merging {ds_name}...")
            with open(anno_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Build a local category mapping for this dataset
            local_cat_id_to_unified_id = {}
            for cat in data.get("categories", []):
                local_cat_id_to_unified_id[cat["id"]] = get_unified_category_id(cat["name"])

            # Map old image ID to new unified image ID
            local_img_id_to_unified_id = {}
            for img in data.get("images", []):
                old_file_name = img["file_name"]
                new_file_name = f"{ds_name}_{old_file_name}"

                # Copy image
                src_img_path = split_path / old_file_name
                dst_img_path = UNIFIED_DIR / split / "images" / new_file_name
                if src_img_path.exists() and not dst_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)

                # Add to unified images
                unified_img = img.copy()
                unified_img["id"] = image_id_counter
                unified_img["file_name"] = new_file_name
                unified_coco["images"].append(unified_img)

                local_img_id_to_unified_id[img["id"]] = image_id_counter
                image_id_counter += 1

            # Map annotations
            for ann in data.get("annotations", []):
                if ann["image_id"] not in local_img_id_to_unified_id:
                    continue
                unified_ann = ann.copy()
                unified_ann["id"] = annotation_id_counter
                unified_ann["image_id"] = local_img_id_to_unified_id[ann["image_id"]]
                unified_ann["category_id"] = local_cat_id_to_unified_id[ann["category_id"]]
                unified_coco["annotations"].append(unified_ann)
                annotation_id_counter += 1

        # Save unified annotations
        out_anno_path = UNIFIED_DIR / split / "annotations" / "_annotations.coco.json"
        with open(out_anno_path, "w", encoding="utf-8") as f:
            json.dump(unified_coco, f)
        print(f"Saved {split} unified annotations to {out_anno_path}\n")

if __name__ == "__main__":
    main()
