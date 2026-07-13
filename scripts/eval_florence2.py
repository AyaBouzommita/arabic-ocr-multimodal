"""CLI: evaluate Florence-2 mAP and inference speed."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.evaluator import evaluate_florence2


def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence-2 detection candidate")
    parser.add_argument("--model-dir", default="results/florence2/model")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics-path", default="results/florence2/metrics.json")
    args = parser.parse_args()

    config = Florence2Config(
        model_dir=Path(args.model_dir),
        metrics_path=Path(args.metrics_path),
    )
    metrics = evaluate_florence2(
        model_dir=Path(args.model_dir),
        split=args.split,
        limit=args.limit,
        config=config,
    )

    print("=" * 60)
    print("  FLORENCE-2 EVALUATION")
    print("=" * 60)
    print(f"Split:             {metrics['split']}")
    print(f"Images evaluated:  {metrics['eval_images']}")
    print(f"mAP@0.5:           {metrics['map50']}")
    print(f"mAP@0.5:0.95:      {metrics['map50_95']}")
    print(f"Avg speed:         {metrics['avg_inference_ms']} ms/image")
    print(f"Metrics saved:     {metrics['metrics_path']}")


if __name__ == "__main__":
    main()
