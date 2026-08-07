from typing import List, Any
import torch

from ocr.postprocessing.arabert_corrector import AraBERTCorrector
from ocr.postprocessing.roberta_corrector import RoBERTaCorrector

class BilingualCorrector:
    """Orchestrates AraBERT and RoBERTa to correct mixed Arabic-English OCR text."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("[BilingualCorrector] Initializing dual-language models...")
        self.arabert = AraBERTCorrector(device=self.device)
        self.roberta = RoBERTaCorrector(device=self.device)
        
    def correct_tokens(
        self,
        tokens: List[Any],
        conf_threshold: float = 90.0,
        max_edit_dist: int = 1
    ) -> str:
        """
        Runs English correction followed by Arabic correction on the token list.
        """
        # Flatten tokens by splitting spaces, so MLM models receive single words natively
        class MockToken:
            def __init__(self, text, conf):
                self.text = text
                self.confidence = conf

        flattened_tokens = []
        for t in tokens:
            subwords = t.text.split()
            # If a token has multiple words, we assign the same confidence to all subwords
            for sw in subwords:
                flattened_tokens.append(MockToken(sw, t.confidence))

        # 1. RoBERTa modifies the English words, returns a list of words
        english_corrected_words = self.roberta.correct_tokens(
            flattened_tokens, conf_threshold=conf_threshold, max_edit_dist=max_edit_dist
        )
        # We preserve the original confidence for unchanged words, 
        # and set confidence to 100.0 for words RoBERTa already corrected so AraBERT ignores them
        original_words = [t.text for t in flattened_tokens]
        original_confs = [t.confidence for t in flattened_tokens]
        
        mock_tokens = []
        for i, word in enumerate(english_corrected_words):
            if word != original_words[i]:
                # RoBERTa corrected this, lock it
                mock_tokens.append(MockToken(word, 100.0))
            else:
                mock_tokens.append(MockToken(word, original_confs[i]))
                
        # 2. AraBERT modifies the Arabic words, returns a final string
        final_text = self.arabert.correct_tokens(
            mock_tokens, conf_threshold=conf_threshold, max_edit_dist=max_edit_dist
        )
        
        return final_text

    def correct_text(self, text, *args, **kwargs):
        # Fallback if raw string is passed
        return self.arabert.correct_text(text, *args, **kwargs)
