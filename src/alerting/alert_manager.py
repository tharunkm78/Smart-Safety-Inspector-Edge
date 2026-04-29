"""Central alert dispatcher — routes detections to audio, visual, and log outputs.

Alert rules:
- CRITICAL → immediate audio (beep_sequence) + rapid visual blink + log
- HIGH     → audio beep + log
- MEDIUM   → visual blink only + log
- LOW      → log only
- Alert cooldown: per-class, configurable in ALERT_COOLDOWNS
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.alerting.alert_logger import log_alert
from src.alerting.audio_alert import AudioAlert
from src.alerting.visual_alert import VisualAlert
from src.config import ALERT_COOLDOWNS


@dataclass
class AlertEvent:
    """A single alert event to be dispatched."""
    class_name: str
    class_id: int
    confidence: float
    priority: str
    bbox: list
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AlertManager:
    """Central alert dispatcher with cooldown tracking."""

    def __init__(self):
        self._audio = AudioAlert()
        self._visual = VisualAlert()

        # Per-class cooldowns: class_name → last_alert_time
        self._last_alert: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        self._active = True

    def dispatch(self, detections: list[dict]):
        """Process a list of detections and trigger appropriate alerts.

        Args:
            detections: List of detection dicts from YOLOv8nDetector.detect()
        """
        if not detections:
            self._set_ok_status()
            return

        # Find highest priority in this frame
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        highest = min(detections, key=lambda d: priority_order.get(d["priority"], 99))
        highest_priority = highest["priority"]

        # Check cooldown for each detection
        now = time.time()
        alerted = []

        with self._lock:
            for det in detections:
                cls = det["class"]
                cooldown = ALERT_COOLDOWNS.get(highest_priority, 5)
                if now - self._last_alert[cls] < cooldown:
                    continue

                self._last_alert[cls] = now
                alerted.append(det)

                # Log to database
                log_alert(
                    class_name=det["class"],
                    class_id=det["class_id"],
                    confidence=det["confidence"],
                    priority=det["priority"],
                    bbox=det["bbox"],
                )

        if not alerted:
            return

        # Dispatch alerts based on highest priority in frame
        self._dispatch_by_priority(highest_priority, alerted)

    def _dispatch_by_priority(self, priority: str, detections: list[dict]):
        """Send alerts to audio/visual/log based on priority."""
        if priority == "CRITICAL":
            self._audio.beep_sequence("CRITICAL", count=5)
            for _ in range(3):
                self._visual.trigger("CRITICAL")
                time.sleep(0.05)
        elif priority == "HIGH":
            self._audio.beep_sequence("HIGH", count=3)
            self._visual.trigger("HIGH")
        elif priority == "MEDIUM":
            self._visual.trigger("MEDIUM")

    def _set_ok_status(self):
        """Set LEDs to green OK state when no hazards detected."""
        self._visual.set_status("OK")

    def stop(self):
        """Stop the alert manager."""
        self._active = False
        self._set_ok_status()
