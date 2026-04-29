"""Visual alert driver — LED patterns by priority.

Jetson: GPIO via RPi.GPIO library
Windows: console output + GUI indicator (tkinter)
"""

import sys
import time
from pathlib import Path

from src.config import is_jetson, is_windows, JETSON_GPIO_PINS


class VisualAlert:
    """Controls LEDs (and buzzer relay) based on hazard priority."""

    # LED blink intervals by priority (seconds)
    BLINK_INTERVALS = {
        "CRITICAL": 0.15,
        "HIGH": 0.3,
        "MEDIUM": 0.6,
        "LOW": 1.0,
        "OK": 2.0,
    }

    def __init__(self):
        self.platform = "jetson" if is_jetson() else ("windows" if is_windows() else "linux")
        self._gpio = None
        self._tk_root = None
        self._led_canvas = None

        if self.platform == "jetson":
            self._init_jetson_gpio()
        else:
            self._init_windows_indicator()

    def _init_jetson_gpio(self):
        """Initialize GPIO on Jetson Orin Nano."""
        try:
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BOARD)

            # Set up all pins as outputs
            for name, pin in JESON_GPIO_PINS.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
                print(f"[VisualAlert] GPIO pin {pin} ({name}) initialized")
            self._gpio = GPIO
        except ImportError:
            print("[VisualAlert] Jetson.GPIO not available — using simulated output")
        except Exception as e:
            print(f"[VisualAlert] GPIO init error: {e}")

    def _init_windows_indicator(self):
        """Initialize a small tkinter window as visual alert indicator on Windows."""
        try:
            import tkinter as tk
            self._tk_root = tk.Tk()
            self._tk_root.title("Safety Alert")
            self._tk_root.geometry("120x60")
            self._tk_root.resizable(False, False)

            self._led_canvas = tk.Canvas(self._tk_root, width=120, height=60, bg="#1a1a1a")
            self._led_canvas.pack()

            # Circle indicator
            self._led_circle = self._led_canvas.create_oval(35, 15, 85, 55, fill="#222222", outline="#444")
            self._led_label = self._led_canvas.create_text(60, 35, text="OK", fill="#00ff00", font=("Arial", 12, "bold"))

            self._tk_root.after(100, self._tk_update)
            self._tk_root.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent close
        except Exception as e:
            print(f"[VisualAlert] tkinter indicator error: {e}")
            self._tk_root = None

    def _tk_update(self):
        """Keep tkinter event loop running."""
        if self._tk_root:
            self._tk_root.after(100, self._tk_update)

    def _set_gpio(self, led_name: str, state: bool):
        """Set a GPIO pin to HIGH (off) or LOW (on) for a named LED."""
        if self._gpio is None:
            return
        pin = JESON_GPIO_PINS.get(led_name)
        if pin:
            import Jetson.GPIO as GPIO
            self._gpio.output(pin, GPIO.LOW if state else GPIO.HIGH)

    def trigger(self, priority: str):
        """Trigger a visual alert pattern for the given priority."""
        if priority == "OK":
            self._show_ok()
        elif priority == "CRITICAL":
            self._blink_critical()
        elif priority == "HIGH":
            self._blink_high()
        elif priority == "MEDIUM":
            self._blink_medium()
        else:
            self._show_ok()

    def _all_off(self):
        """Turn all LEDs off."""
        if self._gpio:
            for name in ["LED_RED", "LED_YELLOW", "LED_GREEN"]:
                self._set_gpio(name, False)

    def _show_ok(self):
        """Green LED steady on."""
        self._all_off()
        if self._gpio:
            self._set_gpio("LED_GREEN", True)
        else:
            self._update_windows_indicator("#00ff00", "OK")

    def _blink_critical(self):
        """Rapid red LED blink (CRITICAL)."""
        if self._gpio:
            self._set_gpio("LED_RED", True)
            time.sleep(0.15)
            self._set_gpio("LED_RED", False)
        else:
            self._update_windows_indicator("#ff0000", "CRITICAL!")

    def _blink_high(self):
        """Medium yellow LED blink (HIGH)."""
        if self._gpio:
            self._set_gpio("LED_YELLOW", True)
            time.sleep(0.3)
            self._set_gpio("LED_YELLOW", False)
        else:
            self._update_windows_indicator("#ffaa00", "WARNING!")

    def _blink_medium(self):
        """Slow yellow LED blink (MEDIUM)."""
        if self._gpio:
            self._set_gpio("LED_YELLOW", True)
            time.sleep(0.6)
            self._set_gpio("LED_YELLOW", False)
        else:
            self._update_windows_indicator("#ffff00", "CAUTION")

    def _update_windows_indicator(self, color: str, text: str):
        """Update the tkinter LED indicator on Windows."""
        if self._led_canvas:
            try:
                self._led_canvas.itemconfig(self._led_circle, fill=color)
                self._led_canvas.itemconfig(self._led_label, text=text)
            except Exception:
                pass  # Window may have been closed

    def set_status(self, priority: str):
        """Set steady LED state (no blink) based on current highest hazard."""
        self._all_off()
        if priority == "CRITICAL" and self._gpio:
            self._set_gpio("LED_RED", True)
        elif priority == "HIGH" and self._gpio:
            self._set_gpio("LED_YELLOW", True)
        else:
            self._show_ok()
