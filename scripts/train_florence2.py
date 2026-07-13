"""CLI: fine-tune Florence-2 on the shared detection dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.trainer import Florence2Trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Florence-2 for object detection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Limit samples per split for quick runs")
    parser.add_argument("--output-dir", default="results/florence2/model")
    args = parser.parse_args()

    config = Florence2Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_dir=Path(args.output_dir),
    )
    trainer = Florence2Trainer(config=config)
    summary = trainer.train(limit_per_split=args.limit_per_split, epochs=args.epochs)

    print("=" * 60)
    print("  FLORENCE-2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Train samples: {summary['train_samples']}")
    print(f"Val samples:   {summary['val_samples']}")
    print(f"Epochs:        {summary['epochs']}")
    print(f"Device:        {summary['device']}")
    print(f"Train losses:  {summary['history']['train_loss']}")
    print(f"Val losses:    {summary['history']['val_loss']}")
    print(f"Model saved:   {summary['model_dir']}")


if __name__ == "__main__":
    main()
