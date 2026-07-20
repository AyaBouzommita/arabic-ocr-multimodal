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
    epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 5e-6
    max_new_tokens: int = 1024
    num_beams: int = 3
    max_image_size: int = 768
    max_seq_length: int = 256
    max_boxes_per_image: int = 8
    low_vram_threshold_gb: float = 6.0
    force_low_vram: bool = False
    force_full_precision: bool = False
    dataset_yaml: Path = field(default_factory=lambda: project_root() / "data" / "vision" / "dataset.yaml")
    annotations_dir: Path = field(default_factory=lambda: project_root() / "data" / "vision" / "florence_annotations")
    model_dir: Path = field(
        default_factory=lambda: project_root() / "results" / "florence2" / "evaluation" / "florence_comparison" / "florence2"
    )
    metrics_path: Path = field(
        default_factory=lambda: project_root()
        / "results"
        / "florence2"
        / "evaluation"
        / "florence_comparison"
        / "florence2"
        / "metrics.json"
    )
    predictions_dir: Path = field(
        default_factory=lambda: project_root()
        / "results"
        / "florence2"
        / "evaluation"
        / "florence_comparison"
        / "florence2"
        / "predictions"
    )
    reports_dir: Path = field(default_factory=lambda: project_root() / "reports")

    def apply_low_vram_defaults(self) -> None:
        """Tighten limits for GPUs with <= 6 GB VRAM."""
        self.max_image_size = 384
        self.max_seq_length = 128
        self.max_boxes_per_image = 5
        self.lora_r = 4
        self.lora_alpha = 4
        self.num_beams = 1
        self.max_new_tokens = 512
