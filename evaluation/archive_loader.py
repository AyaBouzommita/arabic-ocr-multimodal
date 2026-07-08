"""Loader for Supervisely-format OCR archive datasets.

Expected layout (e.g. archive (2)/Documents/Documents/):
    {Category}/
        img/   → 001.jpg, 002.png, ...
        ann/   → 001.json, 002.json, ...

Ground truth is extracted from annotation JSON files by concatenating
all non-empty "Transcription" tag values on annotated objects.

Usage:
    loader = ArchiveDatasetLoader("path/to/archive (2)")
    pairs = loader.get_pairs()  # (image_path, gt_text, category)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ArchiveDatasetLoader:
    """Load images and ground truth from a Supervisely OCR archive."""

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    def __init__(
        self,
        archive_path: str,
        categories: Optional[List[str]] = None,
    ):
        """Initialise the archive loader.

        Args:
            archive_path: Path to the archive root or Documents/Documents folder.
            categories: Optional list of category folder names to include.
                If None, all categories with img/ and ann/ are included.
        """
        self.archive_path = Path(archive_path)
        self.dataset_root = self._resolve_dataset_root(self.archive_path)
        self.categories = categories

    @staticmethod
    def _resolve_dataset_root(archive_path: Path) -> Path:
        """Find the folder that contains category subdirectories."""
        candidates = [
            archive_path / "Documents" / "Documents",
            archive_path,
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            for child in candidate.iterdir():
                if child.is_dir() and (child / "img").is_dir() and (child / "ann").is_dir():
                    return candidate
        raise FileNotFoundError(
            f"No Supervisely dataset found under: {archive_path}. "
            "Expected category folders with img/ and ann/ subdirectories."
        )

    def list_categories(self) -> List[str]:
        """List category folder names in the dataset."""
        return sorted(
            d.name
            for d in self.dataset_root.iterdir()
            if d.is_dir() and (d / "img").is_dir() and (d / "ann").is_dir()
        )

    def _selected_categories(self) -> List[Path]:
        """Return category directories to scan."""
        available = {
            d.name: d
            for d in self.dataset_root.iterdir()
            if d.is_dir() and (d / "img").is_dir() and (d / "ann").is_dir()
        }
        if self.categories:
            missing = [c for c in self.categories if c not in available]
            if missing:
                raise ValueError(
                    f"Unknown categories: {missing}. "
                    f"Available: {sorted(available.keys())}"
                )
            return [available[c] for c in self.categories]
        return sorted(available.values(), key=lambda p: p.name)

    def list_images(self) -> List[Path]:
        """List all image files across selected categories."""
        images = []
        for category_dir in self._selected_categories():
            img_dir = category_dir / "img"
            images.extend(
                f
                for f in sorted(img_dir.iterdir())
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
            )
        return images

    @staticmethod
    def extract_transcription(annotation_path: Path) -> Optional[str]:
        """Extract ground-truth text from a Supervisely annotation JSON file."""
        if not annotation_path.exists():
            return None

        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        parts = []
        for obj in data.get("objects", []):
            for tag in obj.get("tags", []):
                if tag.get("name") == "Transcription":
                    value = (tag.get("value") or "").strip()
                    if value:
                        parts.append(value)
        if not parts:
            return None
        return " ".join(parts)

    def _annotation_path(self, image_path: Path) -> Path:
        """Resolve the annotation JSON path for an image."""
        category_dir = image_path.parent.parent
        return category_dir / "ann" / f"{image_path.stem}.json"

    def document_id(self, image_path: Path) -> str:
        """Build a stable document id: Category/filename_stem."""
        category = image_path.parent.parent.name
        return f"{category}/{image_path.stem}"

    def load_ground_truth(self, image_path: Path) -> Optional[str]:
        """Load ground truth for an image from its annotation JSON."""
        return self.extract_transcription(self._annotation_path(image_path))

    def get_pairs(
        self, require_ground_truth: bool = False
    ) -> List[Tuple[Path, Optional[str], str]]:
        """Get (image_path, ground_truth, category) tuples.

        Args:
            require_ground_truth: If True, skip images without transcription tags.

        Returns:
            List of (image_path, gt_text_or_None, category) tuples.
        """
        pairs = []
        for category_dir in self._selected_categories():
            category = category_dir.name
            img_dir = category_dir / "img"
            for image_path in sorted(img_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self.SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                gt_text = self.load_ground_truth(image_path)
                if require_ground_truth and not gt_text:
                    continue
                pairs.append((image_path, gt_text, category))
        return pairs

    def summary(self) -> Dict[str, int]:
        """Return corpus counts: total images, with GT, without GT, categories."""
        pairs = self.get_pairs()
        with_gt = sum(1 for _, gt, _ in pairs if gt)
        categories = self.list_categories() if not self.categories else self.categories
        return {
            "categories": len(categories),
            "total_images": len(pairs),
            "with_ground_truth": with_gt,
            "without_ground_truth": len(pairs) - with_gt,
        }
