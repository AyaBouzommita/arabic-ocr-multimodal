"""Candidate Ranker using Language Model Probability Scores and Character Edit Distance.

Ranks OCR replacement candidates based on a multi-factor scoring function:
Score = alpha * log_prob - beta * edit_distance + gamma * (ocr_confidence / 100)
"""

import math
from typing import List, Tuple, Dict, Any


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class CandidateRanker:
    """Language Model Candidate Ranker for Arabic OCR Post-Processing."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.5, gamma: float = 0.5):
        """Initialize candidate ranker with scoring weights.

        Args:
            alpha: Weight for Language Model log-probability score.
            beta: Penalty weight for Levenshtein edit distance.
            gamma: Weight for original OCR confidence score.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def score_candidate(
        self,
        candidate_word: str,
        ocr_word: str,
        lm_probability: float,
        ocr_confidence: float = 50.0,
    ) -> float:
        """Compute candidate ranking score.

        Score = alpha * log(prob) - beta * edit_dist + gamma * (conf / 100)
        """
        # Avoid log(0) by clipping probability
        prob_clipped = max(lm_probability, 1e-7)
        log_prob = math.log(prob_clipped)

        edit_dist = levenshtein_distance(ocr_word, candidate_word)
        norm_conf = ocr_confidence / 100.0

        score = (self.alpha * log_prob) - (self.beta * edit_dist) + (self.gamma * norm_conf)
        return score

    def rank_candidates(
        self,
        candidates: List[Tuple[str, float]],
        ocr_word: str,
        ocr_confidence: float = 50.0,
        max_edit_dist: int = 1,
    ) -> List[Dict[str, Any]]:
        """Rank a list of (candidate_word, lm_probability) tuples.

        Returns sorted list of candidate dicts with calculated scores.
        """
        ranked = []
        for word, prob in candidates:
            dist = levenshtein_distance(ocr_word, word)
            if dist > max_edit_dist:
                continue

            score = self.score_candidate(word, ocr_word, prob, ocr_confidence)
            ranked.append({
                "word": word,
                "probability": prob,
                "edit_distance": dist,
                "score": score,
            })

        # Sort descending by ranking score
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


if __name__ == "__main__":
    # Quick sanity test
    ranker = CandidateRanker()
    ocr_token = "حياك"
    candidates = [("حياة", 0.35), ("حياك", 0.40), ("حياتك", 0.05), ("ثم", 0.15)]
    results = ranker.rank_candidates(candidates, ocr_token, ocr_confidence=60.0)

    print(f"Ranking candidates for OCR word '{ocr_token}':")
    for r in results:
        print(f"  Word: {r['word']:<8} Score: {r['score']:>8.4f} | Prob: {r['probability']:.4f} | EditDist: {r['edit_distance']}")
