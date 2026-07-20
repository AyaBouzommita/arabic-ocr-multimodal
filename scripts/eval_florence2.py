"""CLI: evaluate Florence-2 mAP and inference speed."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.cuda_utils import check_cuda
from arabic_ocr_platform.pipeline.vision.florence2.evaluator import evaluate_florence2


def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence-2 detection candidate")
    parser.add_argument(
        "--model-dir",
        default="results/florence2/evaluation/florence_comparison/florence2",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--metrics-path",
        default="results/florence2/evaluation/florence_comparison/florence2/metrics.json",
    )
    args = parser.parse_args()

    check_cuda(exit_on_failure=False)

    model_dir = Path(args.model_dir)
    config = Florence2Config(
        model_dir=model_dir,
        metrics_path=Path(args.metrics_path),
        predictions_dir=model_dir / "predictions",
    )
    metrics = evaluate_florence2(
        model_dir=model_dir,
        split=args.split,
        limit=args.limit,
        config=config,
    )

    print("\n" + "=" * 60)
    print("  FLORENCE-2 EVALUATION")
    print("=" * 60)
    print(f"Split:             {metrics['split']}")
    print(f"Images evaluated:  {metrics['eval_images']}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"mAP50:             {metrics['map50']:.4f}")
    print(f"mAP50-95:          {metrics['map50_95']:.4f}")
    print(f"Avg speed:         {metrics['avg_inference_ms']} ms/image")
    print(f"Metrics saved:     {metrics['metrics_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
