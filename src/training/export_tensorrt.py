"""Export trained YOLOv8n model to TensorRT for Jetson Orin Nano.

Usage:
    python -m src.training.export_tensorrt
    python -m src.training.export_tensorrt --model models/yolov8n_safety_v1.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MODEL_PT, MODEL_TRT, MODEL_TORCHSCRIPT, is_jetson


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n to TensorRT")
    parser.add_argument("--model", type=Path, default=MODEL_PT, help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"[ERROR] Model not found: {args.model}")
        print("Run training first: python -m src.training.train_yolov8")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"=== Exporting Model to TensorRT ===")
    print(f"Model: {args.model}")
    print(f"Target: {MODEL_TRT}")

    model = YOLO(str(args.model))

    if is_jetson():
        # Jetson: export to TensorRT FP16
        print("\nDetected Jetson — exporting TensorRT FP16...")
        export_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,        # FP16 for TensorRT
            device=0,
        )
    else:
        # Windows/Linux dev: export to TensorRT (requires TensorRT pip package)
        # Falls back to TorchScript for CPU/GPU testing
        print("\nNon-Jetson platform — exporting TorchScript (portable)...")
        try:
            export_path = model.export(format="engine", imgsz=args.imgsz, half=True)
        except Exception as e:
            print(f"  TensorRT export failed ({e}), falling back to TorchScript...")
            export_path = model.export(format="torchscript", imgsz=args.imgsz)

    export_path = Path(export_path)
    if export_path.exists() and export_path != MODEL_TRT:
        import shutil
        shutil.copy2(export_path, MODEL_TRT)
        print(f"Copied to: {MODEL_TRT}")

    print("\n=== Export Complete ===")
    print(f"Output: {MODEL_TRT if MODEL_TRT.exists() else export_path}")


if __name__ == "__main__":
    main()
