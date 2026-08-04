from dataclasses import dataclass

@dataclass
class NLPConfig:
    # We use MARBERTv2 which is optimized for Masked Language Modeling on Arabic text
    model_name: str = "UBC-NLP/MARBERTv2"
    # OCR confidence threshold below which we apply NLP correction
    correction_threshold: float = 0.8
    # Number of candidates to generate per masked word
    top_k_candidates: int = 3
    # Use GPU if available
    device: int = -1  # -1 for CPU, 0 for GPU
