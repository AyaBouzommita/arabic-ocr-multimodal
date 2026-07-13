"""CLI: full Florence-2 bake-off pipeline (train + eval + reports)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arabic_ocr_platform.pipeline.vision.florence2.bakeoff import (
    generate_bakeoff_comparison,
    generate_florence_candidate_report,
)
from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.evaluator import evaluate_florence2
from arabic_ocr_platform.pipeline.vision.florence2.trainer import Florence2Trainer


def main():
    parser = argparse.ArgumentParser(description="Run full Florence-2 bake-off pipeline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--model-dir", default="results/florence2/model")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = Florence2Config(epochs=args.epochs, model_dir=Path(args.model_dir))

    train_summary = None
    if not args.skip_train:
        print("⏳ Training Florence-2...")
        trainer = Florence2Trainer(config=config)
        train_summary = trainer.train(limit_per_split=args.train_limit, epochs=args.epochs)

    print("⏳ Evaluating Florence-2...")
    metrics = evaluate_florence2(
        model_dir=Path(args.model_dir),
        split=args.split,
        limit=args.eval_limit,
        config=config,
    )

    if train_summary:
        metrics["train_samples"] = train_summary["train_samples"]
        metrics["val_samples"] = train_summary["val_samples"]
        metrics["epochs"] = train_summary["epochs"]
        metrics["device"] = train_summary["device"]
        metrics["final_train_loss"] = train_summary["history"]["train_loss"][-1]
        metrics["final_val_loss"] = train_summary["history"]["val_loss"][-1]

    reports_dir = config.reports_dir
    candidate_report = reports_dir / "florence2_candidate.md"
    comparison_report = reports_dir / "vision_bakeoff_comparison.md"

    print("⏳ Generating reports...")
    generate_florence_candidate_report(metrics, candidate_report)
    generate_bakeoff_comparison(
        florence_metrics=metrics,
        output_path=comparison_report,
        yolo_metrics_path=PROJECT_ROOT / "results" / "yolov8" / "metrics.json",
        detectron_metrics_path=PROJECT_ROOT / "results" / "detectron2" / "metrics.json",
    )

    print("=" * 60)
    print("  FLORENCE-2 BAKE-OFF PIPELINE COMPLETE")
    print("=" * 60)
    print(f"mAP@0.5:      {metrics['map50']}")
    print(f"Speed:        {metrics['avg_inference_ms']} ms/image")
    print(f"Metrics:      {config.metrics_path}")
    print(f"Report:       {candidate_report}")
    print(f"Comparison:   {comparison_report}")


if __name__ == "__main__":
    main()
