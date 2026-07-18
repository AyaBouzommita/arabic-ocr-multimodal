"""
Visualisation des détections Detectron2
Samar Zaabouti - Sprint 2

Ce script prend quelques images du dataset de validation et affiche
les boîtes détectées par Detectron2 dessus, pour voir visuellement
si le modèle détecte bien les éléments du document.
"""

import os
import json
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DATASET_COCO = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_coco"
IMAGES_VAL   = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_for_object_detection/dataset_for_object_detection/valid/images"
OUTPUT_DIR   = "/mnt/c/Users/pc/Documents/stage-neoledge/output_detectron2"
VIZ_DIR      = "/mnt/c/Users/pc/Documents/stage-neoledge/output_detectron2/visualizations"
NUM_IMAGES   = 10  # Nombre d'images à visualiser

CLASSES = [
    "text", "table", "picture", "signature", "stamp",
    "qr_code", "3en", "5aaa", "5en", "aleph", "baaa",
    "daal", "dad", "faaa", "geem", "haaa", "hamza",
    "hamzasater", "kaaf", "lam", "mem", "non", "qaf",
    "raaa", "sad", "sen", "sheen", "taaa", "thaa",
    "thal", "then", "ttaa", "waaa", "yaaa", "zaaa", "letters"
]

# ----------------------------------------------------------------------
# ENREGISTREMENT DU DATASET
# ----------------------------------------------------------------------
for name in ["arabic_docs_val"]:
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)

register_coco_instances("arabic_docs_val", {}, f"{DATASET_COCO}/val.json", IMAGES_VAL)
MetadataCatalog.get("arabic_docs_val").thing_classes = CLASSES

# ----------------------------------------------------------------------
# CONFIGURATION DU MODÈLE
# ----------------------------------------------------------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
cfg.MODEL.DEVICE  = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 36
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # seuil bas pour voir plus de détections

predictor = DefaultPredictor(cfg)
metadata  = MetadataCatalog.get("arabic_docs_val")

# ----------------------------------------------------------------------
# VISUALISATION
# ----------------------------------------------------------------------
os.makedirs(VIZ_DIR, exist_ok=True)

# Récupère les images de validation
with open(f"{DATASET_COCO}/val.json", 'r') as f:
    val_data = json.load(f)

images = val_data['images'][:NUM_IMAGES]
print(f"Visualisation de {len(images)} images...\n")

for i, img_info in enumerate(images):
    img_path = os.path.join(IMAGES_VAL, img_info['file_name'])

    if not os.path.exists(img_path):
        print(f"⚠️  Image introuvable : {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Prédiction
    outputs = predictor(img)
    instances = outputs["instances"]

    print(f"[{i+1}/{len(images)}] {img_info['file_name']}")
    print(f"  → {len(instances)} détections trouvées")

    if len(instances) > 0:
        classes_detected = [CLASSES[c] for c in instances.pred_classes.tolist()]
        scores = [f"{s:.2f}" for s in instances.scores.tolist()]
        for cls, score in zip(classes_detected, scores):
            print(f"     {cls} ({score})")

    # Visualisation avec Detectron2 Visualizer
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    v = Visualizer(img_rgb, metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(instances.to("cpu"))
    result_img = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)

    # Sauvegarde
    out_path = os.path.join(VIZ_DIR, f"detection_{i+1:02d}_{img_info['file_name']}")
    cv2.imwrite(out_path, result_img)
    print(f"  → Sauvegardé : {out_path}\n")

print(f"\n✅ Terminé ! Images dans : {VIZ_DIR}")
print(f"Ouvre ce dossier dans l'Explorateur Windows pour voir les résultats.")
