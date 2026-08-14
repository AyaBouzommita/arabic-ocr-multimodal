import cv2
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TrOCRRecognizer:
    """Microsoft TrOCR-based text recognizer for printed Latin/French text.
    
    Uses a Vision Transformer encoder + language model decoder for
    significantly better character-level accuracy compared to CRNN-based
    engines like EasyOCR, especially for accented characters.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        logger.info(f"[TrOCR] Loading model '{model_name}' on {self.device}...")
        print(f"[TrOCR] Loading model '{model_name}' on {self.device}...")
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("[TrOCR] Model loaded successfully!")
            logger.info("[TrOCR] Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load TrOCR model '{model_name}': {e}")
            raise
    
    def recognize_crop(self, crop_bgr: np.ndarray, max_new_tokens: int = 128) -> str:
        """Recognize text from a BGR image crop.
        
        Args:
            crop_bgr: OpenCV BGR image numpy array of a text region crop.
            max_new_tokens: Maximum number of tokens to generate.
            
        Returns:
            Recognized text string. Returns empty string if crop is invalid.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        
        try:
            # Convert BGR (OpenCV) to RGB PIL Image
            if len(crop_bgr.shape) == 2:  # Grayscale
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
            else:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(crop_rgb)
            
            # Process image for the model
            pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,  # Beam search for better accuracy
                    early_stopping=True,
                )
            
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            logger.error(f"Error during recognition of crop: {e}")
            return ""
    
    def recognize_crops_batch(self, crops: List[np.ndarray], max_new_tokens: int = 128, batch_size: int = 8) -> List[str]:
        """Recognize text from multiple BGR image crops in batches.
        
        Args:
            crops: List of OpenCV BGR image numpy arrays.
            max_new_tokens: Maximum number of tokens to generate per crop.
            batch_size: Number of images to process at once.
            
        Returns:
            List of recognized text strings. Elements are empty strings for invalid crops.
        """
        results = []
        
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_pil = []
            valid_indices = []
            
            for j, crop_bgr in enumerate(batch_crops):
                if crop_bgr is None or crop_bgr.size == 0:
                    # Provide a dummy image to maintain batch consistency if needed, 
                    # but we can handle it later or by providing a white image.
                    batch_pil.append(Image.new('RGB', (100, 32), color='white'))
                else:
                    try:
                        if len(crop_bgr.shape) == 2:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
                        else:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        batch_pil.append(Image.fromarray(crop_rgb))
                        valid_indices.append(j)
                    except Exception as e:
                        logger.error(f"Error converting crop at batch index {j}: {e}")
                        batch_pil.append(Image.new('RGB', (100, 32), color='white'))
            
            try:
                pixel_values = self.processor(images=batch_pil, return_tensors="pt", padding=True).pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_new_tokens=max_new_tokens,
                        num_beams=4,
                        early_stopping=True,
                    )
                
                texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Clear text for completely invalid crops
                batch_results = []
                for j, text in enumerate(texts):
                    if j in valid_indices:
                        batch_results.append(text.strip())
                    else:
                        batch_results.append("")
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error during batch recognition: {e}")
                # Append empty strings on failure to match batch size
                results.extend(["" for _ in batch_crops])
        
        return results
