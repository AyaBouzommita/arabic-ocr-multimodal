from django.apps import AppConfig
import torch
import os

class OcrApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ocr_api'
    
    # Global model instances
    yolo_model = None
    paddleocr_ar = None
    paddleocr_fr = None
    # Keep old names as aliases for backward compatibility
    easyocr_engine = None
    easyocr_fr = None
    enhancer = None
    bilingual_corrector = None

    def ready(self):
        # The ready method runs when Django starts. 
        # We load models here to ensure they stay in memory.
        
        # In dev mode, Django's autoreloader runs ready() twice. We can prevent double loading:
        if os.environ.get('RUN_MAIN', None) != 'true':
            # Skip if this is the autoreloader thread
            return

        print("\n[Django] Loading OCR Pipeline Models into memory...")
        
        try:
            from ultralytics import YOLO
            from paddleocr import PaddleOCR
            from ocr.easyocr.engine import EasyOCREngine
            from ocr.postprocessing.bilingual_corrector import BilingualCorrector
            from ocr.preprocessing.image_enhancer import ImageEnhancer

            # 1. YOLO
            print("[Django] Loading YOLO...")
            from django.conf import settings
            yolo_path = os.path.join(settings.BASE_DIR.parent, "runs", "detect", "evaluation", "yolo_comparison", "yolov11s_finetuned", "weights", "best.pt")
            if os.path.exists(yolo_path):
                OcrApiConfig.yolo_model = YOLO(yolo_path)
            else:
                print(f"[Django] WARNING: YOLO model not found at {yolo_path}. Falling back to standard yolo11s.pt!")
                OcrApiConfig.yolo_model = YOLO("yolo11s.pt")
            
            # 2. Hybrid OCR: EasyOCR for Arabic/Mixed, PaddleOCR v4 for pure French/Latin
            use_gpu = torch.cuda.is_available()
            
            print("[Django] Loading EasyOCR (Arabic/English)...")
            OcrApiConfig.easyocr_engine = EasyOCREngine(
                languages=['ar', 'en'], 
                gpu=use_gpu, 
                paragraph=True
            )
            
            print("[Django] Loading PaddleOCR v4 (Arabic - ensemble partner)...")
            OcrApiConfig.paddleocr_ar = PaddleOCR(
                lang='ar',
                use_angle_cls=True,
                use_gpu=use_gpu,
                ocr_version='PP-OCRv4',
                show_log=False,
            )
            
            print("[Django] Loading PaddleOCR v4 (French)...")
            OcrApiConfig.paddleocr_fr = PaddleOCR(
                lang='fr',
                use_angle_cls=True,
                use_gpu=use_gpu,
                ocr_version='PP-OCRv4',
                show_log=False,
            )
            
            # Set aliases so views.py backward-compat check works
            OcrApiConfig.easyocr_fr = OcrApiConfig.paddleocr_fr
            
            # 3. Image Enhancer
            print("[Django] Loading Image Enhancer...")
            OcrApiConfig.enhancer = ImageEnhancer()
            
            # 4. AraBERT / RoBERTa (Bilingual Corrector)
            print("[Django] Loading AraBERT & RoBERTa...")
            OcrApiConfig.bilingual_corrector = BilingualCorrector(device="cuda" if torch.cuda.is_available() else "cpu")
            
            print("[Django] All models loaded successfully! Server is ready.\n")
            
        except Exception as e:
            print(f"[Django] Error loading models: {e}")
            import traceback
            traceback.print_exc()

