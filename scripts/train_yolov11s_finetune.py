"""Fine-tune trained YOLOv11s model on 300-sample dataset with a low learning rate."""

import torch
from ultralytics import YOLO

BASE_WEIGHTS = "runs/detect/evaluation/yolo_comparison/yolov11s/weights/best.pt"
DATASET_YAML = "C:/Users/zaabola/Desktop/neoledge-OCR/data/finetune_dataset/dataset.yaml"

def check_cuda():
    print("=" * 60)
    print("  CUDA / GPU Check for Fine-Tuning")
    print("=" * 60)
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device:       {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ERROR: CUDA is NOT available!")
        exit(1)
    print("=" * 60)

def main():
    check_cuda()

    print(f"\n[Loading Base Model] {BASE_WEIGHTS}...")
    model = YOLO(BASE_WEIGHTS)

    print("\n[Fine-Tuning] Running YOLOv11s fine-tuning (epochs=25, lr0=0.001)...")
    results = model.train(
        data=DATASET_YAML,
        epochs=25,
        imgsz=640,
        batch=16,
        lr0=0.001,      # Low learning rate for fine-tuning (10x lower than initial 0.01)
        lrf=0.01,       # Final learning rate fraction
        device=0,       # Force GPU
        project="evaluation/yolo_comparison",
        name="yolov11s_finetuned",
        patience=10,
        save=True,
        plots=True,
    )

    print("\n[Evaluating] Validating fine-tuned model...")
    metrics = model.val(data=DATASET_YAML, split="val", device=0)

    print("\n" + "=" * 60)
    print("  YOLOv11s Fine-Tuning Complete!")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
