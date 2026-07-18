"""
Sprint 2 - Detectron2 Training Script
Samar Zaabouti - NeoLedge Stage 2026
"""

import os
import json

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

DATASET_COCO = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_coco"
IMAGES_TRAIN = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_for_object_detection/dataset_for_object_detection/train/images"
IMAGES_VAL   = "/mnt/c/Users/pc/Documents/stage-neoledge/dataset_for_object_detection/dataset_for_object_detection/valid/images"
OUTPUT_DIR   = "/mnt/c/Users/pc/Documents/stage-neoledge/output_detectron2"

for name in ["arabic_docs_train", "arabic_docs_val"]:
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)

register_coco_instances("arabic_docs_train", {}, f"{DATASET_COCO}/train.json", IMAGES_TRAIN)
register_coco_instances("arabic_docs_val",   {}, f"{DATASET_COCO}/val.json",   IMAGES_VAL)

print("Dataset enregistré ✅")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("arabic_docs_train",)
cfg.DATASETS.TEST  = ("arabic_docs_val",)
cfg.MODEL.WEIGHTS  = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE                         = "cpu"
cfg.SOLVER.IMS_PER_BATCH                 = 2
cfg.SOLVER.BASE_LR                       = 0.00025
cfg.SOLVER.MAX_ITER                      = 500
cfg.SOLVER.STEPS                         = (350, 450)
cfg.SOLVER.CHECKPOINT_PERIOD             = 250
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 36
cfg.TEST.EVAL_PERIOD                     = 100
cfg.OUTPUT_DIR                           = OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n=== Entraînement Detectron2 ===")
print(f"Iterations : {cfg.SOLVER.MAX_ITER}")
print(f"Output     : {OUTPUT_DIR}\n")

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("\n✅ Entraînement terminé !")

cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor  = DefaultPredictor(cfg)
evaluator  = COCOEvaluator("arabic_docs_val", output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "arabic_docs_val")
results    = inference_on_dataset(predictor.model, val_loader, evaluator)

print("\n=== Résultats ===")
print(results)

with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📄 Résultats : {OUTPUT_DIR}/evaluation_results.json")
print(f"📦 Modèle    : {OUTPUT_DIR}/model_final.pth")
