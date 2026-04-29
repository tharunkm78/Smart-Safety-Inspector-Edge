"""Core YOLOv8n detector with PyTorch and TensorRT inference.

Usage:
    python -m src.inference.detector --image tests/test.jpg
    python -m src.inference.detector --camera 0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    MODEL_PT,
    MODEL_TRT,
    SAFETY_CLASSES,
    CLASS_TO_IDX,
    CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    is_jetson,
)


class YOLOv8nDetector:
    """YOLOv8n safety detector with TensorRT and PyTorch backends."""

    def __init__(self, model_path: Optional[Path] = None, conf_thresh: float = CONFIDENCE_THRESHOLD):
        self.conf_thresh = conf_thresh

        # Resolve model path
        model_path = model_path or (MODEL_TRT if is_jetson() else MODEL_PT)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        from ultralytics import YOLO
        import torch
        
        # Select device
        if is_jetson():
            self.device = 0
        elif torch.cuda.is_available():
            self.device = 0
            print(f"SUCCESS: Using GPU ({torch.cuda.get_device_name(0)}) for inference.")
        else:
            self.device = "cpu"
            print("WARNING: Using CPU for inference. GPU not detected or not compatible.")

        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.model_path = model_path
        self.inference_times = []

    def detect(self, frame, imgsz: int = 640) -> list[dict]:
        """Run detection on a single frame.

        Returns:
            List of detections: [{"class": str, "class_id": int, "confidence": float,
                                  "bbox": [x1, y1, x2, y2], "priority": str}, ...]
        """
        from src.inference.postprocess import PostProcessor
        from src.config import ALERT_PRIORITY

        t0 = time.perf_counter()
        results = self.model(
            frame,
            imgsz=imgsz,
            conf=self.conf_thresh,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
            device=self.device,
        )
        t1 = time.perf_counter()
        self.inference_times.append(t1 - t0)

        detections = []
        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            cls_name = SAFETY_CLASSES[cls_id] if cls_id < len(SAFETY_CLASSES) else f"class_{cls_id}"
            priority = ALERT_PRIORITY.get(cls_name, "LOW")

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": [round(x, 2) for x in xyxy],
                "priority": priority,
            })

        return detections

    def avg_fps(self) -> float:
        if not self.inference_times:
            return 0.0
        return 1.0 / (sum(self.inference_times) / len(self.inference_times))

    def reset_timers(self):
        self.inference_times = []


def run_image(detector: YOLOv8nDetector, image_path: Path):
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    detections = detector.detect(img)
    print(f"\nImage: {image_path}")
    print(f"Detections ({len(detections)}):")
    for d in detections:
        print(f"  [{d['priority']:8s}] {d['class']:20s} {d['confidence']:.2%}  bbox={d['bbox']}")

    # Draw boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = {"CRITICAL": (0, 0, 255), "HIGH": (0, 165, 255), "MEDIUM": (0, 255, 255), "LOW": (0, 255, 0)}[det["priority"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")


def run_camera(detector: YOLOv8nDetector, camera_idx: int = 0):
    import cv2
    from src.inference.camera import Camera

    cam = Camera(camera_idx=camera_idx)
    cam.start()
    print(f"\nCamera feed active. Press 'q' to quit.")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            detections = detector.detect(frame)

            # Draw on frame
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                color = {"CRITICAL": (0, 0, 255), "HIGH": (0, 165, 255), "MEDIUM": (0, 255, 255), "LOW": (0, 255, 0)}[det["priority"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']} {det['confidence']:.0%}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            fps = detector.avg_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Safety Inspector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8n safety detector")
    parser.add_argument("--image", type=Path, help="Path to test image")
    parser.add_argument("--camera", type=int, default=None, help="Camera index to use")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    print("=== YOLOv8n Safety Detector ===")
    model_path = args.model or (MODEL_TRT if is_jetson() else MODEL_PT)
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {args.conf}")

    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Run training first: python -m src.training.train_yolov8")
        sys.exit(1)

    detector = YOLOv8nDetector(model_path=model_path, conf_thresh=args.conf)

    if args.image:
        run_image(detector, args.image)
    elif args.camera is not None:
        run_camera(detector, args.camera)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
