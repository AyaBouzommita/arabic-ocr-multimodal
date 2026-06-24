import re
import jiwer


def normalize_arabic_text(
    text: str,
    remove_diacritics: bool = True,
    normalize_alef_ta_ya: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize Arabic text to clean it up for robust CER/WER comparison.

    Args:
        text: Input string.
        remove_diacritics: If True, strips short vowels (harakat) and tatweel.
        normalize_alef_ta_ya: If True, unifies Alef shapes, Teh Marbuta, and Alef Maksura.
        remove_punctuation: If True, strips standard and Arabic punctuation.

    Returns:
        Normalized clean string.
    """
    if not text:
        return ""

    # Remove leading/trailing whitespaces and unify whitespaces
    text = text.strip()

    if remove_diacritics:
        # Arabic diacritics unicode range
        # Fatha, Damma, Kasra, Sukun, Shadda, Tanweens, Tatweel (kashida)
        diacritics_pattern = re.compile(r"[\u064B-\u0652\u0640]")
        text = diacritics_pattern.sub("", text)

    if normalize_alef_ta_ya:
        # Unify Alef (أ, إ, آ -> ا)
        text = re.sub(r"[أإآ]", "ا", text)
        # Unify Teh Marbuta (ة -> ه)
        text = re.sub(r"ة", "ه", text)
        # Unify Alef Maksura (ى -> ي)
        text = re.sub(r"ى", "ي", text)

    if remove_punctuation:
        # Punctuation: Latin and Arabic (، ؛ ؟)
        punctuation_pattern = re.compile(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\'،؛؟]')
        text = punctuation_pattern.sub(" ", text)

    # Replace multiple whitespaces/tabs/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_cer(
    reference: str, hypothesis: str, normalize: bool = True, **norm_kwargs
) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis text.

    Handling edge cases:
        - If reference and hypothesis are both empty: returns 0.0.
        - If reference is empty but hypothesis is not: returns 1.0.
        - If hypothesis is empty but reference is not: returns 1.0.

    Args:
        reference: The ground truth string.
        hypothesis: The OCR predicted string.
        normalize: If True, applies Arabic normalization.
        norm_kwargs: Additional arguments for normalize_arabic_text.

    Returns:
        CER value as a float (typically between 0.0 and 1.0, though can exceed 1.0 due to insertions).
    """
    if normalize:
        ref_clean = normalize_arabic_text(reference, **norm_kwargs)
        hyp_clean = normalize_arabic_text(hypothesis, **norm_kwargs)
    else:
        ref_clean = reference.strip() if reference else ""
        hyp_clean = hypothesis.strip() if hypothesis else ""

    # Edge case handling
    if not ref_clean and not hyp_clean:
        return 0.0
    if not ref_clean or not hyp_clean:
        return 1.0

    try:
        return float(jiwer.cer(ref_clean, hyp_clean))
    except Exception:
        # Fallback manual calculation if jiwer fails
        return _levenshtein_distance(ref_clean, hyp_clean) / len(ref_clean)


def calculate_wer(
    reference: str, hypothesis: str, normalize: bool = True, **norm_kwargs
) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis text.

    Handling edge cases:
        - If reference and hypothesis are both empty: returns 0.0.
        - If reference is empty but hypothesis is not: returns 1.0.
        - If hypothesis is empty but reference is not: returns 1.0.

    Args:
        reference: The ground truth string.
        hypothesis: The OCR predicted string.
        normalize: If True, applies Arabic normalization.
        norm_kwargs: Additional arguments for normalize_arabic_text.

    Returns:
        WER value as a float.
    """
    if normalize:
        ref_clean = normalize_arabic_text(reference, **norm_kwargs)
        hyp_clean = normalize_arabic_text(hypothesis, **norm_kwargs)
    else:
        ref_clean = reference.strip() if reference else ""
        hyp_clean = hypothesis.strip() if hypothesis else ""

    # Edge case handling
    if not ref_clean and not hyp_clean:
        return 0.0
    if not ref_clean or not hyp_clean:
        return 1.0

    try:
        return float(jiwer.wer(ref_clean, hyp_clean))
    except Exception:
        # Fallback manual word calculation if jiwer fails
        ref_words = ref_clean.split()
        hyp_words = hyp_clean.split()
        if not ref_words and not hyp_words:
            return 0.0
        if not ref_words or not hyp_words:
            return 1.0
        return _levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def _levenshtein_distance(seq1, seq2) -> int:
    """Levenshtein distance calculation fallback for sequences (strings or lists of words)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + 1,  # Substitution
                )
    return dp[m][n]
