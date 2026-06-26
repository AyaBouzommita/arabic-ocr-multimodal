"""Ground truth loader for the CER/WER evaluation pipeline.

Provides utilities to load ground-truth text files and pair them
with their corresponding document images. This is Ilyess's half
of US-07 — the loader infrastructure that feeds into Aya's scorer.

Convention:
    For each image  data/raw/foo.png
    Ground truth is data/ground_truth/foo.txt

Usage:
    loader = GroundTruthLoader("data/raw", "data/ground_truth")
    pairs = loader.get_pairs()
    for image_path, gt_text in pairs:
        print(f"{image_path}: {gt_text[:50]}...")
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GroundTruthLoader:
    """Load and pair ground-truth text with document images.

    Scans the image directory and ground-truth directory, matching
    files by stem name (e.g., sample.png ↔ sample.txt).

    Attributes:
        image_dir: Path to the directory containing document images.
        gt_dir: Path to the directory containing ground-truth text files.
        supported_extensions: Set of supported image file extensions.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    def __init__(
        self,
        image_dir: str = "data/raw",
        gt_dir: str = "data/ground_truth",
    ):
        """Initialize the ground truth loader.

        Args:
            image_dir: Path to the directory containing document images.
            gt_dir: Path to the directory containing ground-truth .txt files.
        """
        self.image_dir = Path(image_dir)
        self.gt_dir = Path(gt_dir)

    def list_images(self) -> List[Path]:
        """List all image files in the image directory.

        Returns:
            Sorted list of image file paths.
        """
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        images = [
            f
            for f in sorted(self.image_dir.iterdir())
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
        ]
        return images

    def load_ground_truth(self, stem: str) -> Optional[str]:
        """Load the ground-truth text for a document by its filename stem.

        Args:
            stem: Filename stem (e.g., 'sample' for sample.png).

        Returns:
            The ground-truth text string, or None if the file doesn't exist.
        """
        gt_path = self.gt_dir / f"{stem}.txt"
        if not gt_path.exists():
            return None

        return gt_path.read_text(encoding="utf-8").strip()

    def get_pairs(self) -> List[Tuple[Path, str]]:
        """Get all (image_path, ground_truth_text) pairs.

        Only returns pairs where both the image and its corresponding
        ground-truth file exist.

        Returns:
            List of (image_path, gt_text) tuples.
        """
        pairs = []
        for image_path in self.list_images():
            gt_text = self.load_ground_truth(image_path.stem)
            if gt_text is not None:
                pairs.append((image_path, gt_text))
        return pairs

    def get_unpaired_images(self) -> List[Path]:
        """Find images that have no corresponding ground-truth file.

        Returns:
            List of image paths without ground truth.
        """
        unpaired = []
        for image_path in self.list_images():
            gt_text = self.load_ground_truth(image_path.stem)
            if gt_text is None:
                unpaired.append(image_path)
        return unpaired

    def summary(self) -> Dict[str, int]:
        """Get a summary of the corpus: total images, paired, unpaired.

        Returns:
            Dictionary with counts: total_images, paired, unpaired.
        """
        all_images = self.list_images()
        pairs = self.get_pairs()
        return {
            "total_images": len(all_images),
            "paired": len(pairs),
            "unpaired": len(all_images) - len(pairs),
        }


def load_corpus_metadata(csv_path: str = "data/corpus_metadata.csv") -> List[Dict]:
    """Load the corpus metadata CSV file.

    The CSV should have columns: filename, document_type, source, has_ground_truth.

    Args:
        csv_path: Path to the corpus metadata CSV file.

    Returns:
        List of metadata dictionaries, one per document.
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
