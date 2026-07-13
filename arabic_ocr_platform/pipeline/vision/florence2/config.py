"""Default configuration for Florence-2 detection bake-off."""

from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


@dataclass
class Florence2Config:
    checkpoint: str = "microsoft/Florence-2-base-ft"
    revision: str = "refs/pr/6"
    task_prompt: str = "<OD>"
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 5e-6
    max_new_tokens: int = 1024
    num_beams: int = 3
    dataset_yaml: Path = field(default_factory=lambda: project_root() / "data" / "vision" / "dataset.yaml")
    annotations_dir: Path = field(default_factory=lambda: project_root() / "data" / "vision" / "florence_annotations")
    model_dir: Path = field(default_factory=lambda: project_root() / "results" / "florence2" / "model")
    metrics_path: Path = field(default_factory=lambda: project_root() / "results" / "florence2" / "metrics.json")
    predictions_dir: Path = field(default_factory=lambda: project_root() / "results" / "florence2" / "predictions")
    reports_dir: Path = field(default_factory=lambda: project_root() / "reports")
