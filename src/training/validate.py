"""Validate trained model mAP on validation set.

Usage:
    python -m src.training.validate
    python -m src.training.validate --model models/yolov8n_safety_v1.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MODEL_PT, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8n model")
    parser.add_argument("--model", type=Path, default=MODEL_PT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"=== Validating Model ===")
    print(f"Model: {args.model}")

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(DATA_DIR / "combined" / "dataset.yaml"),
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=True,
    )

    print(f"\nmAP@0.5:  {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Per-class AP:")
    for i, ap in enumerate(metrics.box.ap50):
        name = model.names.get(i, f"class_{i}")
        print(f"  {name}: {ap:.4f}")


if __name__ == "__main__":
    main()
