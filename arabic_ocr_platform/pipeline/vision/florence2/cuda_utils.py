"""CUDA / GPU checks aligned with YOLO training scripts."""

from __future__ import annotations

import sys

import torch


def check_cuda(exit_on_failure: bool = True) -> bool:
    """Print GPU status in the same format as scripts/train_yolov11s.py."""
    print("=" * 60)
    print("  CUDA / GPU Check")
    print("=" * 60)
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:     {torch.version.cuda}")
        print(f"  GPU device:       {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("=" * 60)
        return True

    print("  ERROR: CUDA is NOT available! Training will be on CPU (very slow).")
    print("  Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 60)
    if exit_on_failure:
        sys.exit(1)
    return False
