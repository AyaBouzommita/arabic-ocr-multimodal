import torch
from ultralytics import YOLO

DATASET = "C:/Users/zaabola/Desktop/neoledge-OCR/data/dataset_for_object_detection/dataset.yaml"

def check_cuda():
    print("=" * 60)
    print("  CUDA / GPU Check")
    print("=" * 60)
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:     {torch.version.cuda}")
        print(f"  GPU device:       {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ERROR: CUDA is NOT available! Training will be on CPU (very slow).")
        print("  Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        exit(1)
    print("=" * 60)

def main():
    check_cuda()

    print("\n  Training YOLOv11m on Arabic Document Layout Dataset\n")

    model = YOLO("yolo11m.pt")

    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,  # Reduced for RTX 3050 Ti (4GB VRAM)
        device=0,  # Force GPU 0
        project="evaluation/yolo_comparison",
        name="yolov11m",
        patience=10,
        save=True,
        plots=True,
    )

    print("\nRunning validation on test set...")
    metrics = model.val(data=DATASET, split="test", device=0)

    print("\n" + "=" * 60)
    print("  YOLOv11m Training Complete!")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
