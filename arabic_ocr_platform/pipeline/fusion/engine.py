import logging

logger = logging.getLogger(__name__)

class FusionEngine:
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha (float): Weight given to OCR confidence. (1 - alpha) is given to LM score.
        """
        self.alpha = alpha

    def _is_inside_bbox(self, token_bbox, object_bbox):
        """Check if token_bbox center is inside object_bbox."""
        if len(token_bbox) != 4 or len(object_bbox) != 4:
            return False
            
        tx1, ty1, tx2, ty2 = token_bbox
        ox1, oy1, ox2, oy2 = object_bbox
        
        cx = (tx1 + tx2) / 2
        cy = (ty1 + ty2) / 2
        
        return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)

    def process(self, ocr_data: dict, detected_objects: list, correction_candidates: list) -> dict:
        """
        Fuses OCR, Vision (detected_objects), and NLP (correction_candidates) data.
        Returns the final structured fusion output matching fusion_input.schema.json.
        """
        logger.info("Fusing OCR, Vision, and NLP candidates...")
        
        # Build candidate lookup dictionary
        # In corrector, we used position [i, i+1] where i is the token index
        candidate_lookup = {}
        for cand in correction_candidates:
            token_idx = cand['position'][0]
            candidate_lookup[token_idx] = cand['candidates']

        fused_tokens = []
        tokens = ocr_data.get('tokens', [])
        
        for i, token in enumerate(tokens):
            final_text = token['text']
            ocr_conf = token.get('confidence', 1.0)
            token_bbox = token.get('bbox', [0,0,0,0])
            
            # 1. Vision Context Rules
            in_stamp_or_logo = False
            for obj in detected_objects:
                if obj['label'] in ['Stamp', 'Logo', 'Signature']:
                    if self._is_inside_bbox(token_bbox, obj['bbox']):
                        in_stamp_or_logo = True
                        break
            
            # If in a stamp/logo, it might be noise or not part of the main text.
            # We could penalize confidence or skip correction.
            if in_stamp_or_logo:
                ocr_conf *= 0.5
                
            # 2. Apply NLP Correction if available
            if i in candidate_lookup and not in_stamp_or_logo:
                best_score = -1.0
                best_cand = None
                
                for cand in candidate_lookup[i]:
                    lm_score = cand['lm_score']
                    # Simple fusion score
                    combined_score = (self.alpha * ocr_conf) + ((1 - self.alpha) * lm_score)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_cand = cand['text']
                
                # If the combined score is better than keeping the original (assuming original score = alpha * ocr_conf)
                # Actually, if we have candidates, we replace if the best candidate is confident.
                if best_cand and best_score > (self.alpha * ocr_conf):
                    logger.debug(f"Replacing '{final_text}' with '{best_cand}' (score: {best_score:.3f})")
                    final_text = best_cand
                    # We might update the token confidence to reflect the new fused confidence,
                    # but schema expects a float 0-1.
                    ocr_conf = min(best_score, 1.0)
            
            fused_tokens.append({
                "text": final_text,
                "bbox": token_bbox,
                "confidence": ocr_conf
            })

        # Construct the full text from tokens
        fused_raw_text = " ".join([t['text'] for t in fused_tokens])
        
        # Build the final fusion schema format
        fusion_output = {
            "document_id": ocr_data.get("document_id", "unknown"),
            "ocr": {
                "document_id": ocr_data.get("document_id", "unknown"),
                "engine": ocr_data.get("engine", "paddleocr"),
                "raw_text": fused_raw_text,
                "tokens": fused_tokens,
                "processing_time_ms": ocr_data.get("processing_time_ms", 0)
            },
            "detected_objects": detected_objects,
            "correction_candidates": correction_candidates
        }
        
        return fusion_output
