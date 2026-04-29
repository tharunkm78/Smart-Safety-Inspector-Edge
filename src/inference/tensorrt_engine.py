"""TensorRT inference wrapper for Jetson Orin Nano.

Provides a lower-level TensorRT runtime interface on top of the YOLOv8n detector,
for maximum performance on the edge device.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import MODEL_TRT, SAFETY_CLASSES, CONFIDENCE_THRESHOLD


class TensorRTEngine:
    """TensorRT runtime wrapper for YOLOv8n engine files."""

    def __init__(self, engine_path: Optional[Path] = None, conf_thresh: float = CONFIDENCE_THRESHOLD):
        self.engine_path = engine_path or MODEL_TRT
        self.conf_thresh = conf_thresh
        self._engine = None
        self._context = None
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._input_shape = (1, 3, 640, 640)

        if self.engine_path and self.engine_path.exists():
            self._load_engine()
        else:
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

    def _load_engine(self):
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as f:
                self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
            print(f"TensorRT engine loaded: {self.engine_path}")
        except ImportError:
            raise RuntimeError("TensorRT not installed. Run: pip install tensorrt")

    def infer(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a frame (HWC uint8 → CHW FP32 normalized).

        Returns list of detections in same format as YOLOv8nDetector.detect().
        """
        import cv2
        from src.config import ALERT_PRIORITY

        # Preprocess: resize to 640x640, normalize to [0, 1], convert to CHW
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Allocate buffers (simplified — real impl uses cudaMemcpy)
        input_tensor = np.ascontiguousarray(img)
        output_buf = np.zeros((1, 8400, 84), dtype=np.float32)

        # Run inference
        self._context.execute_v2(bindings=[int(input_tensor), int(output_buf)])

        # Parse output (YOLOv8n output shape: [1, 8400, 84] = [box_params, class_scores])
        detections = self._parse_outputs(output_buf, frame.shape[:2])
        return detections

    def _parse_outputs(self, output: np.ndarray, orig_shape: tuple) -> list[dict]:
        """Parse YOLOv8n TensorRT output into detection dicts."""
        from src.config import ALERT_PRIORITY

        # output shape: (1, 8400, 84) where 84 = 4(box) + 80(classes)
        # Simplified parsing — actual ultralytics post-processing is more complex
        boxes = output[0, :, :4]
        class_scores = output[0, :, 4:]

        detections = []
        for i in range(boxes.shape[0]):
            x, y, w, h = boxes[i]
            scores = class_scores[i]
            max_score = float(np.max(scores))
            if max_score < self.conf_thresh:
                continue
            cls_id = int(np.argmax(scores))

            # Convert from 640x640 to original shape
            x1 = (x - w / 2) / 640 * orig_shape[1]
            y1 = (y - h / 2) / 640 * orig_shape[0]
            x2 = (x + w / 2) / 640 * orig_shape[1]
            y2 = (y + h / 2) / 640 * orig_shape[0]

            cls_name = SAFETY_CLASSES[cls_id] if cls_id < len(SAFETY_CLASSES) else f"class_{cls_id}"
            priority = ALERT_PRIORITY.get(cls_name, "LOW")

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(max_score, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "priority": priority,
            })

        return detections
