"""CLI: fine-tune Florence-2 on the shared detection dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.cuda_utils import check_cuda
from arabic_ocr_platform.pipeline.vision.florence2.evaluator import evaluate_florence2
from arabic_ocr_platform.pipeline.vision.florence2.trainer import Florence2Trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Florence-2 for object detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Limit samples per split for quick runs")
    parser.add_argument(
        "--output-dir",
        default="results/florence2/evaluation/florence_comparison/florence2",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip test-set evaluation after training")
    args = parser.parse_args()

    check_cuda()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.json"

    print("\n  Training Florence-2 on Arabic Document Layout Dataset\n")

    config = Florence2Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_dir=output_dir,
        metrics_path=metrics_path,
        predictions_dir=output_dir / "predictions",
    )
    trainer = Florence2Trainer(config=config)
    summary = trainer.train(limit_per_split=args.limit_per_split, epochs=args.epochs)

    print("\n" + "=" * 60)
    print("  FLORENCE-2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Train samples: {summary['train_samples']}")
    print(f"Val samples:   {summary['val_samples']}")
    print(f"Epochs:        {summary['epochs']}")
    print(f"Device:        {summary['device']}")
    print(f"Model saved:   {summary['model_dir']}")

    if args.skip_eval:
        return

    print("\nRunning validation on test set...")
    metrics = evaluate_florence2(
        model_dir=output_dir,
        split="test",
        limit=args.limit_per_split,
        config=config,
    )

    print("\n" + "=" * 60)
    print("  Florence-2 Training Complete!")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  mAP50:     {metrics['map50']:.4f}")
    print(f"  mAP50-95:  {metrics['map50_95']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
