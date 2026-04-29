"""Train YOLOv8n nano on the combined safety dataset.

Usage:
    python -m src.training.train_yolov8
    python -m src.training.train_yolov8 --epochs 50 --batch 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, MODEL_PT, TRAIN_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on safety dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt to resume from")
    args = parser.parse_args()

    from ultralytics import YOLO

    print("=== YOLOv8n Safety Model Training ===\n")
    print(f"Dataset: {DATA_DIR / 'combined'}")
    print(f"Model save: {MODEL_PT}")

    cfg = TRAIN_CONFIG.copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.device is not None:
        cfg["device"] = args.device
    cfg["imgsz"] = args.imgsz

    print(f"Config: {cfg}\n")

    # Load pretrained model
    model = YOLO(cfg["model"])

    # Device selection
    device = cfg.get("device")
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    # Train the model
    results = model.train(
        data=str(DATA_DIR / "combined" / "dataset.yaml"),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=device,
        augment=cfg["augment"],
        mosaic=cfg["mosaic"],
        copy_paste=cfg["copy_paste"],
        fliplr=cfg["fliplr"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        cls=cfg["cls"],
        workers=cfg["workers"],
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=cfg["exist_ok"],
        pretrained=cfg["pretrained"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        patience=cfg["patience"],
        resume=args.resume,
    )

    # Save best model to expected path
    best_pt = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, MODEL_PT)
        print(f"\nBest model copied to: {MODEL_PT}")

    print("\n=== Training Complete ===")
    print(f"Results: {results.save_dir}")


if __name__ == "__main__":
    main()
