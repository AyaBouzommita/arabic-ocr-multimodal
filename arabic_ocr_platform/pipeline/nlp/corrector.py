import time
import logging
from transformers import pipeline

from arabic_ocr_platform.pipeline.nlp.config import NLPConfig

logger = logging.getLogger(__name__)

class NLPCorrector:
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        logger.info(f"Loading NLP Corrector with model {self.config.model_name}")
        
        # Load the fill-mask pipeline for MARBERT
        self.fill_mask = pipeline(
            "fill-mask",
            model=self.config.model_name,
            device=self.config.device,
            top_k=self.config.top_k_candidates
        )
        self.mask_token = self.fill_mask.tokenizer.mask_token

    def _build_context_sentence(self, tokens, mask_index):
        """Build a sentence context by concatenating tokens, masking the target index."""
        words = []
        for i, token in enumerate(tokens):
            if i == mask_index:
                words.append(self.mask_token)
            else:
                words.append(token['text'])
        return " ".join(words)

    def process(self, ocr_data: dict) -> list:
        """
        Process OCR data and generate correction candidates for low-confidence words.
        Returns a list of correction candidates matching the fusion_input schema.
        """
        correction_candidates = []
        tokens = ocr_data.get('tokens', [])
        
        for i, token in enumerate(tokens):
            confidence = token.get('confidence', 1.0)
            original_word = token.get('text', '')
            
            # Skip if confidence is high enough
            if confidence >= self.config.correction_threshold:
                continue
                
            # Create masked context
            masked_sentence = self._build_context_sentence(tokens, i)
            
            try:
                # Generate candidates
                predictions = self.fill_mask(masked_sentence)
                
                # Handle single result or list of results
                if not isinstance(predictions, list):
                    predictions = [predictions]
                    
                candidates = []
                for pred in predictions:
                    # Clean up the predicted token (MARBERT sometimes returns tokens with '##')
                    pred_text = pred['token_str'].replace('##', '').strip()
                    
                    # Filter out candidates that are only punctuation
                    # Including standard punctuation and Arabic specific ones like ellipsis(…), comma(،), question mark(؟)
                    import string
                    is_punct = all(char in string.punctuation or char in '…،؛؟' for char in pred_text)
                    
                    if pred_text and not is_punct:
                        candidates.append({
                            "text": pred_text,
                            "lm_score": float(pred['score'])
                        })
                
                if candidates:
                    correction_candidates.append({
                        "original_word": original_word,
                        # position in character indices based on raw_text could be complex,
                        # but schema says "[start, end]". We'll use word indices or approximate char indices.
                        # For simplicity in this demo schema, using word index [i, i+1]
                        "position": [i, i + 1],
                        "candidates": candidates
                    })
                    
            except Exception as e:
                logger.error(f"Error generating candidates for word '{original_word}': {e}")
                
        return correction_candidates
