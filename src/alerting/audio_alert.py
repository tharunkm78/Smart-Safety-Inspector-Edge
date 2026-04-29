"""Audio alert driver — cross-platform buzzer/speaker alerts.

Windows: winsound (built-in)
Jetson/Linux: pygame (or ossaudiodev)
"""

import sys
from pathlib import Path

from src.config import is_jetson, is_windows


class AudioAlert:
    """Plays alert tones based on priority level."""

    FREQUENCIES = {
        "CRITICAL": 880,   # High-pitched warning tone
        "HIGH": 660,        # Medium-high
        "MEDIUM": 440,      # Medium
        "LOW": 220,         # Low beep
    }

    def __init__(self):
        self.platform = "jetson" if is_jetson() else ("windows" if is_windows() else "linux")
        self._initialized = False

    def _init_pygame(self):
        if self._initialized:
            return
        try:
            import pygame
            pygame.mixer.init()
            self._pygame = pygame
            self._initialized = True
        except Exception as e:
            print(f"[AudioAlert] pygame init failed: {e}")
            self._pygame = None
            self._initialized = True

    def _generate_tone(self, frequency: int, duration_ms: int = 500) -> bytes:
        """Generate a simple sine wave tone as WAV bytes."""
        import math
        import struct
        import wave

        import numpy as np

        sample_rate = 22050
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)
        signal = (signal * 32767).astype(np.int16)

        # Write to WAV in memory
        import io
        buf = io.BytesIO()
        with wave.open(buf, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(signal.tobytes())
        return buf.getvalue()

    def play(self, priority: str):
        """Play an alert tone for the given priority level."""
        freq = self.FREQUENCIES.get(priority, 440)
        duration = {"CRITICAL": 800, "HIGH": 500, "MEDIUM": 300, "LOW": 200}.get(priority, 300)

        if self.platform == "windows":
            self._play_windows(freq, duration)
        else:
            self._play_pygame(freq, duration)

    def _play_windows(self, frequency: int, duration_ms: int):
        """Play tone on Windows using winsound."""
        try:
            import winsound
            winsound.Beep(frequency, min(duration_ms, 1000))
        except Exception as e:
            print(f"[AudioAlert] winsound error: {e}")

    def _play_pygame(self, frequency: int, duration_ms: int):
        """Play tone on Jetson/Linux using pygame."""
        self._init_pygame()
        if self._pygame is None:
            print(f"[AudioAlert] Audio not available (pygame failed)")
            return

        try:
            import pygame
            import io
            import numpy as np
            import wave

            sample_rate = 22050
            n_samples = int(sample_rate * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
            signal = np.sin(2 * np.pi * frequency * t)
            signal = (signal * 0.5 * 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(signal.tobytes())
            buf.seek(0)

            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"[AudioAlert] pygame play error: {e}")

    def beep_sequence(self, priority: str, count: int = 3):
        """Play a sequence of beeps."""
        for i in range(count):
            self.play(priority)
            if i < count - 1:
                import time
                time.sleep(0.2)
