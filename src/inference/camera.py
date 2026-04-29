"""Cross-platform camera capture.

Windows: DirectShow via OpenCV
Jetson: V4L2 via OpenCV
"""

import threading
from typing import Optional

import cv2
import numpy as np

from src.config import CAMERA_CONFIG, is_jetson, is_windows


class Camera:
    """Thread-safe camera capture with cross-platform backend selection."""

    def __init__(self, camera_idx: int = 0, width: int = None, height: int = None, fps: int = None):
        self.camera_idx = camera_idx
        self.width = width or CAMERA_CONFIG["width"]
        self.height = height or CAMERA_CONFIG["height"]
        self.fps = fps or CAMERA_CONFIG["fps"]
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._current_frame = None
        self._lock = threading.Lock()

    def _select_backend(self) -> int:
        """Select OpenCV backend based on platform."""
        if is_jetson():
            # V4L2 for Jetson
            return cv2.CAP_V4L2
        else:
            # DirectShow on Windows, default on others
            return cv2.CAP_DSHOW if is_windows() else cv2.CAP_ANY

    def start(self):
        """Start camera capture in background thread."""
        if self._running:
            return

        backend = self._select_backend()
        self._cap = cv2.VideoCapture(self.camera_idx, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Camera {self.camera_idx} could not be opened")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Background loop to continuously grab frames."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._current_frame = frame.copy()
            else:
                with self._lock:
                    self._current_frame = None

    def read(self) -> Optional[np.ndarray]:
        """Get the latest frame (non-blocking)."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
