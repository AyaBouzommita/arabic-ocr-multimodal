"""OCR Noise Filter — detects and removes garbage text from decorative borders and patterns.

Uses statistical heuristics (entropy, special character ratio, character repetition,
vowel analysis) to identify text that is clearly OCR noise rather than real document content.

IMPORTANT: This filter must be conservative — it's better to keep some noise than to
accidentally delete real document content. Only flag text as garbage when we're very
confident it's not real.
"""

import math
import re
import string
from collections import Counter
from typing import List, Any


class OCRNoiseFilter:
    """Filters out garbage OCR text generated from decorative borders, stamps, and patterns.
    
    Design philosophy: Only filter text that is OBVIOUSLY garbage. Long tokens with
    recognizable words must always be preserved, even if they contain some noise.
    """

    def __init__(
        self,
        min_length: int = 2,
        max_special_ratio: float = 0.40,
        min_entropy: float = 1.2,
        max_entropy: float = 5.5,
    ):
        self.min_length = min_length
        self.max_special_ratio = max_special_ratio
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy of a text string."""
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())

    @staticmethod
    def _count_real_words(text: str) -> int:
        """Count space-separated words that look like real language content.
        
        A 'real word' must be a space-separated token consisting of 3+ consecutive
        letters from a single script (Arabic or Latin), with no special chars mixed in.
        """
        real_count = 0
        for word in text.split():
            # Strip common punctuation from edges
            clean = word.strip('.,;:!?()[]{}"\'-/|')
            if not clean:
                continue
            # Check if the word is purely Arabic (3+ chars)
            if re.fullmatch(r'[\u0600-\u06FF]{3,}', clean):
                real_count += 1
            # Check if the word is purely Latin (3+ chars, allows accented)
            elif re.fullmatch(r"[a-zA-ZéèêëàâäôöûüùïîçÉÈÊËÀÂÄÔÖÛÜÙÏÎÇ]{3,}", clean):
                real_count += 1
        return real_count

    def is_garbage(self, text: str, confidence: float = 1.0) -> bool:
        """Determine if a text string is OCR noise/garbage.

        Args:
            text: The OCR-extracted text to evaluate.
            confidence: OCR confidence score (0.0-1.0 scale).

        Returns:
            True if the text is likely garbage/noise, False if it appears to be real text.
        """
        clean_text = text.strip()

        # Rule 0: Too short to be meaningful
        if len(clean_text) < self.min_length:
            return True

        # RESCUE: If the token contains 2+ recognizable words, it's ALWAYS real content.
        # This prevents false positives on bilingual text, mixed-format exam papers, etc.
        real_word_count = self._count_real_words(clean_text)
        if real_word_count >= 2:
            return False

        # Rule 1: Very low OCR confidence
        if confidence < 0.30:
            return True

        # Rule 2: Special character ratio (excludes alphanumeric, spaces, Arabic, French accented)
        non_alpha = re.sub(
            r"[\w\s\u0600-\u06FF'éèêëàâäôöûüùïîçÉÈÊËÀÂÄÔÖÛÜÙÏÎÇ°/:.,;×!?()\-\"]",
            '', clean_text
        )
        if len(clean_text) > 0:
            special_ratio = len(non_alpha) / len(clean_text)
            if special_ratio > self.max_special_ratio:
                return True

        # Rule 3: Repeated character sequences (e.g., "####", "aaaa", "8888")
        if re.search(r'(.)\1{4,}', clean_text):
            return True

        # Rule 4: Low unique character ratio for longer strings (decorative pattern indicator)
        if len(clean_text) > 10:
            unique_ratio = len(set(clean_text)) / len(clean_text)
            if unique_ratio < 0.18:
                return True

        # Rule 5: Shannon entropy check — only for SHORT tokens (<=30 chars)
        # Long tokens naturally have varied entropy and should be handled by the rescue rule above
        if 6 < len(clean_text) <= 30:
            entropy = self.shannon_entropy(clean_text)
            if entropy < self.min_entropy or entropy > self.max_entropy:
                return True

        # Rule 6: Hash/special chars interspersed within word-like tokens
        # e.g., "#4f#AAaARA8#8" — special chars mixed into what should be a word
        words = clean_text.split()
        if words:
            garbage_word_count = 0
            for w in words:
                if re.search(r'[#&%@${}|\\]', w):
                    garbage_word_count += 1
            # Only flag if MOST words contain special chars
            if len(words) <= 3 and garbage_word_count / len(words) > 0.5:
                return True

        # Rule 7: Digit-only noise — text that's PURELY digits with no letter content at all
        # e.g., "8 787478" — but preserve things like "6x1=6" or "N/08"
        digits = re.findall(r'\d', clean_text)
        all_chars = re.findall(r'\S', clean_text)
        if len(all_chars) > 3 and len(digits) / len(all_chars) > 0.80:
            if real_word_count == 0:
                return True

        return False

    def filter_tokens(self, tokens: List[Any]) -> List[Any]:
        """Filter a list of OCR tokens, removing garbage entries.

        Args:
            tokens: List of OCRToken objects with .text and .confidence attributes.

        Returns:
            Filtered list with garbage tokens removed.
        """
        filtered = []
        for token in tokens:
            conf = token.confidence / 100.0 if token.confidence > 1.0 else token.confidence
            if not self.is_garbage(token.text, confidence=conf):
                filtered.append(token)
        return filtered
