from ultralytics import YOLO

DATASET = "C:/Users/zaabola/Desktop/neoledge-OCR/data/dataset_for_object_detection/dataset.yaml"

def main():
    model = YOLO("yolov8n.pt")
    print("Starting YOLOv8n training...")
    
    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=16,
        project="evaluation",
        name="yolo_layout_model",
    )
    
    print("YOLOv8n training completed. Check 'evaluation/yolo_layout_model' for results.")

if __name__ == "__main__":
    main()
