"""Post-processing utilities for detection outputs."""

from typing import NamedTuple

import numpy as np

from src.config import SAFETY_CLASSES, ALERT_PRIORITY


class Detection(NamedTuple):
    """Single detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    priority: str

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
            "priority": self.priority,
        }


class PostProcessor:
    """Handles NMS, confidence filtering, and class mapping."""

    def __init__(self, conf_thresh: float = 0.35, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def filter_by_confidence(self, detections: list[dict]) -> list[dict]:
        """Remove detections below confidence threshold."""
        return [d for d in detections if d["confidence"] >= self.conf_thresh]

    def apply_nms(self, detections: list[dict]) -> list[dict]:
        """Apply Non-Maximum Suppression to remove overlapping boxes of same class."""
        if not detections:
            return []

        # Group by class
        from collections import defaultdict
        by_class = defaultdict(list)
        for d in detections:
            by_class[d["class"]].append(d)

        kept = []
        for cls_name, cls_dets in by_class.items():
            # Sort by confidence descending
            cls_dets = sorted(cls_dets, key=lambda x: x["confidence"], reverse=True)

            for i, det in enumerate(cls_dets):
                keep = True
                for other in cls_dets[:i]:
                    if self._iou(det["bbox"], other["bbox"]) > self.iou_thresh:
                        keep = False
                        break
                if keep:
                    kept.append(det)

        return kept

    def _iou(self, box_a: list, box_b: list) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def highest_priority(self, detections: list[dict]) -> str:
        """Return the highest alert priority among detections."""
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        if not detections:
            return "OK"
        return min(detections, key=lambda d: priority_order.get(d["priority"], 99))["priority"]

    def to_detection_objs(self, raw_dets: list[dict]) -> list[Detection]:
        """Convert raw detection dicts to Detection NamedTuples."""
        results = []
        for d in self.filter_by_confidence(raw_dets):
            results.append(Detection(
                class_name=d["class"],
                class_id=d["class_id"],
                confidence=d["confidence"],
                bbox=tuple(d["bbox"]),
                priority=d["priority"],
            ))
        return results
