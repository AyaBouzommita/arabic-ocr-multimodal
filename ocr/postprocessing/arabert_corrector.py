"""AraBERT Contextual OCR Corrector using Masked Language Modeling (MLM).

Includes Smart Confidence-Gated correction to preserve high-confidence OCR predictions
and strict 1-character edit distance filtering to eliminate hallucinated words.
"""

import torch
import string
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple, Optional, Dict, Any, Union

from spellchecker import SpellChecker

from ocr.postprocessing.candidate_ranker import CandidateRanker, levenshtein_distance

MODEL_NAME = "aubmindlab/bert-base-arabertv02"


class AraBERTCorrector:
    """Smart Contextual Arabic OCR Error Corrector powered by AraBERT & CandidateRanker."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        ranker: Optional[CandidateRanker] = None,
    ):
        """Initialize AraBERT tokenizer, Masked LM model, and CandidateRanker."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spell = SpellChecker(language='ar')
        print(f"[AraBERT] Loading tokenizer and model '{model_name}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.ranker = ranker or CandidateRanker(alpha=1.0, beta=1.5, gamma=0.5)

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print("[AraBERT] Smart model & CandidateRanker loaded successfully!")

    def is_arabic_word(self, word: str) -> bool:
        """Check if word is predominantly Arabic."""
        eng_chars = sum(1 for c in word if c in string.ascii_letters)
        arab_chars = sum(1 for c in word if "\u0600" <= c <= "\u06FF")
        
        # If it has both, it's mixed OCR noise. Classify based on majority.
        if arab_chars > 0 and arab_chars > eng_chars:
            return True
        return False

    def is_valid_in_lexicon(self, word: str) -> bool:
        """Check if the word exists in the Arabic dictionary using pyspellchecker."""
        # Clean non-Arabic chars before checking
        clean_word = "".join(c for c in word if "\u0600" <= c <= "\u06FF")
        if not clean_word:
            return False
            
        # If spell.unknown returns empty, it means the word is known (valid)
        if not self.spell.unknown([clean_word]):
            return True
            
        return False

    def correct_word_in_context(
        self,
        words: List[str],
        target_idx: int,
        ocr_confidence: float = 50.0,
        top_k: int = 15,
        max_edit_dist: int = 1,
    ) -> Tuple[str, bool]:
        """Predict and rank contextual replacements for words[target_idx] using CandidateRanker."""
        original_word = words[target_idx]
        if not self.is_arabic_word(original_word) or len(original_word) < 2:
            return original_word, False

        # Create a sliding window around the target index to prevent 512-token limit errors
        WINDOW_SIZE = 32
        start_idx = max(0, target_idx - WINDOW_SIZE)
        end_idx = min(len(words), target_idx + WINDOW_SIZE + 1)
        
        local_words = words[start_idx:end_idx].copy()
        local_target_idx = target_idx - start_idx
        
        # Build masked sentence for the local window
        local_words[local_target_idx] = self.mask_token
        masked_text = " ".join(local_words)

        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] > 512:
            return original_word, False

        input_ids = inputs["input_ids"][0]
        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            return original_word, False

        mask_pos = mask_positions[0].item()

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, mask_pos]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)

        # Collect candidates from MLM
        candidates = []
        for prob, idx in zip(top_probs, top_indices):
            token_str = self.tokenizer.decode([idx.item()]).strip()
            # Clean BPE artifacts if they somehow survive decode
            token_str = token_str.replace("##", "").strip()

            if token_str and self.is_arabic_word(token_str):
                candidates.append((token_str, prob.item()))
                
        # Add candidates from SpellChecker (with low fallback probability)
        # This helps when the context is so corrupted that MLM fails to predict the right word
        spell_cands = self.spell.candidates(original_word) or set()
        existing_words = [c[0] for c in candidates]
        for cand in spell_cands:
            if cand not in existing_words:
                candidates.append((cand, 0.001))

        if not candidates:
            return original_word, False
        
        # Rank candidates using CandidateRanker
        ranked_results = self.ranker.rank_candidates(
            candidates, original_word, ocr_confidence=ocr_confidence, max_edit_dist=max_edit_dist
        )

        if not ranked_results:
            return original_word, False

        # Filter candidates by lexicon validity (must be a real word)
        valid_candidates = []
        for cand in ranked_results:
            if cand["probability"] > 0.01 and self.is_valid_in_lexicon(cand["word"]):
                valid_candidates.append(cand)
                
        if not valid_candidates:
            return original_word, False

        best_cand = valid_candidates[0]
        if best_cand["word"] != original_word:
            return best_cand["word"], True

        return original_word, False

    def correct_tokens(
        self,
        tokens: List[Any],
        conf_threshold: float = 80.0,
        max_edit_dist: int = 1,
    ) -> str:
        """Smart confidence-gated correction.

        Only modifies tokens with confidence < conf_threshold or containing non-Arabic noise.
        High-confidence OCR predictions are preserved 100% untouched.
        """
        if not tokens:
            return ""

        words = [t.text for t in tokens]
        confs = [t.confidence for t in tokens]

        corrected_words = words.copy()

        for idx in range(len(words)):
            word = words[idx]
            conf = confs[idx]

            if not self.is_arabic_word(word):
                continue

            # Strip English OCR noise characters from the Arabic word
            clean_word = "".join(c for c in word if c not in string.ascii_letters)
            
            # Update word with clean_word in the working list for AraBERT masking
            words[idx] = clean_word
            word = clean_word

            # GUARD 1: If confidence >= conf_threshold, DO NOT touch!
            if conf >= conf_threshold:
                corrected_words[idx] = word
                continue
                
            # GUARD 2: If the word is already a valid Arabic word in the dictionary, DO NOT touch!
            if self.is_valid_in_lexicon(word):
                corrected_words[idx] = word
                continue

            # Otherwise, attempt strict 1-character typo correction
            new_word, changed = self.correct_word_in_context(
                words, idx, top_k=10, max_edit_dist=max_edit_dist
            )
            if changed:
                corrected_words[idx] = new_word
            else:
                corrected_words[idx] = word

        return " ".join(corrected_words)

    def correct_text(
        self,
        text: Union[str, List[Any]],
        top_k: int = 10,
        max_edit_dist: int = 1,
        conf_threshold: float = 90.0,
    ) -> str:
        """Run contextual correction over words in text.
        
        If text is a list of OCRTokens, uses correct_tokens for confidence gating.
        """
        if isinstance(text, list):
            return self.correct_tokens(text, conf_threshold=conf_threshold, max_edit_dist=max_edit_dist)
            
        if not text or not text.strip():
            return text

        words = text.split()
        if len(words) < 2:
            return text

        corrected_words = words.copy()

        for idx in range(len(words)):
            word = words[idx]
            
            # GUARD: Lexicon check even for plain strings
            if self.is_arabic_word(word) and self.is_valid_in_lexicon(word):
                continue
                
            new_word, changed = self.correct_word_in_context(
                words, idx, top_k=top_k, max_edit_dist=max_edit_dist
            )
            if changed:
                corrected_words[idx] = new_word

        return " ".join(corrected_words)
