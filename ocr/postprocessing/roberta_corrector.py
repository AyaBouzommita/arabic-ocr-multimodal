"""RoBERTa Contextual OCR Corrector using Masked Language Modeling (MLM).

Includes Smart Confidence-Gated correction to preserve high-confidence OCR predictions
and strict 1-character edit distance filtering to eliminate hallucinated words.
"""

import torch
import string
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple, Optional, Dict, Any, Union
from spellchecker import SpellChecker

from ocr.postprocessing.candidate_ranker import CandidateRanker, levenshtein_distance

MODEL_NAME = "roberta-base"


class RoBERTaCorrector:
    """Smart Contextual English OCR Error Corrector powered by RoBERTa & CandidateRanker."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        ranker: Optional[CandidateRanker] = None,
    ):
        """Initialize RoBERTa tokenizer, Masked LM model, and CandidateRanker."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spell = SpellChecker(language='en')
        print(f"[RoBERTa] Loading tokenizer and model '{model_name}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.ranker = ranker or CandidateRanker(alpha=1.0, beta=1.5, gamma=0.5)

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print("[RoBERTa] Smart model & CandidateRanker loaded successfully!")

    def is_english_word(self, word: str) -> bool:
        """Check if word is predominantly English."""
        eng_chars = sum(1 for c in word if c in string.ascii_letters)
        arab_chars = sum(1 for c in word if "\u0600" <= c <= "\u06FF")
        
        # If it has both, it's mixed OCR noise. Classify based on majority.
        if eng_chars > 0 and eng_chars >= arab_chars:
            return True
        return False

    def is_valid_in_lexicon(self, word: str) -> bool:
        """Check if the word exists in the English dictionary using pyspellchecker."""
        # Clean punctuation from the word before checking
        clean_word = "".join(c for c in word if c in string.ascii_letters).lower()
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
        top_k: int = 10,
        max_edit_dist: int = 1,
    ) -> Tuple[str, bool]:
        """
        Masks the word at target_idx, queries RoBERTa, and returns the best candidate
        that is within `max_edit_dist` characters of the original OCR text.
        """
        original_word = words[target_idx]

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
            token_str = token_str.replace("Ġ", "").strip()

            if token_str and self.is_english_word(token_str):
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

        # Rank Candidates using CandidateRanker
        ranked_results = self.ranker.rank_candidates(candidates, original_word, max_edit_dist=max_edit_dist)

        print(f"DEBUG [{original_word}]: Ranked results = {ranked_results[:3]}")

        if not ranked_results:
            return original_word, False

        best_cand = ranked_results[0]
        best_candidate = best_cand["word"]

        if best_candidate != original_word:
            # Preserve capitalization of first letter if originally capitalized
            if original_word[0].isupper():
                best_candidate = best_candidate.capitalize()
            return best_candidate, True

        return original_word, False

    def correct_tokens(
        self,
        tokens: List[Any],
        conf_threshold: float = 90.0,
        max_edit_dist: int = 1,
    ) -> List[str]:
        """
        Applies confidence-gated correction to English words.
        Expects `tokens` to have `.text` and `.confidence` attributes.
        Returns a list of strings (corrected words).
        """
        words = [t.text for t in tokens]
        confs = [t.confidence for t in tokens]

        corrected_words = words.copy()

        for idx in range(len(words)):
            word = words[idx]
            conf = confs[idx]

            if not self.is_english_word(word):
                continue

            # Strip Arabic OCR noise characters from the English word
            clean_word = "".join(c for c in word if not ("\u0600" <= c <= "\u06FF"))
            
            # Update word with clean_word in the working list for RoBERTa masking
            words[idx] = clean_word
            word = clean_word

            # GUARD 1: If confidence >= conf_threshold, DO NOT touch!
            if conf >= conf_threshold:
                # If we cleaned it, we still want to save the clean version
                corrected_words[idx] = word
                continue
                
            # GUARD 2: If the word is already a valid English word in the dictionary, DO NOT touch!
            if self.is_valid_in_lexicon(word):
                corrected_words[idx] = word
                continue

            # Otherwise, attempt strict typo correction
            new_word, changed = self.correct_word_in_context(
                words, idx, top_k=10, max_edit_dist=max_edit_dist
            )
            
            if changed:
                corrected_words[idx] = new_word
            else:
                corrected_words[idx] = word

        return corrected_words
