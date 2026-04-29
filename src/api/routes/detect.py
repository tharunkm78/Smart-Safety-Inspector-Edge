"""POST /api/detect — run inference on a base64-encoded image."""

import base64
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from src.inference.detector import YOLOv8nDetector
from src.alerting.alert_manager import AlertManager
from src.config import MODEL_PT, MODEL_TRT, is_jetson

router = APIRouter()


class DetectRequest(BaseModel):
    image: str  # base64 encoded image


class DetectionResponse(BaseModel):
    detections: list
    fps: float
    priority: str


# Global detector instance (lazy-loaded)
_detector: Optional[YOLOv8nDetector] = None
_alert_manager: Optional[AlertManager] = None


def get_detector() -> YOLOv8nDetector:
    global _detector
    if _detector is None:
        model_path = MODEL_TRT if is_jetson() else MODEL_PT
        _detector = YOLOv8nDetector(model_path=model_path)
    return _detector


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


@router.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectRequest):
    """Run safety detection on a base64-encoded image."""
    from PIL import Image
    import cv2
    import numpy as np

    detector = get_detector()
    alert_mgr = get_alert_manager()

    # Decode base64 → image
    img_bytes = base64.b64decode(request.image)
    img_pil = Image.open(BytesIO(img_bytes))
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Run detection
    detections = detector.detect(img)
    fps = detector.avg_fps()

    # Dispatch alerts
    alert_mgr.dispatch(detections)

    # Determine highest priority
    from src.inference.postprocess import PostProcessor
    pp = PostProcessor()
    highest_priority = pp.highest_priority(detections)

    return DetectResponse(
        detections=detections,
        fps=round(fps, 1),
        priority=highest_priority,
    )


@router.post("/detect/file")
async def detect_file(file: UploadFile = File(...)):
    """Run safety detection on an uploaded image file."""
    from PIL import Image
    import cv2
    import numpy as np

    detector = get_detector()
    alert_mgr = get_alert_manager()

    contents = await file.read()
    img_pil = Image.open(BytesIO(contents))
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    detections = detector.detect(img)
    fps = detector.avg_fps()
    alert_mgr.dispatch(detections)

    from src.inference.postprocess import PostProcessor
    pp = PostProcessor()
    highest_priority = pp.highest_priority(detections)

    return {"detections": detections, "fps": round(fps, 1), "priority": highest_priority}
