"""GPIO control for Jetson Orin Nano (L4T / Linux).

Uses Jetson.GPIO library (L4T Python bindings) to control:
  Pin 7  — Red LED   (CRITICAL alert)
  Pin 11 — Yellow LED (WARNING)
  Pin 13 — Green LED  (OK / heartbeat)
  Pin 15 — Buzzer trigger
"""

from src.config import JETSON_GPIO_PINS, is_jetson

if is_jetson():
    import Jetson.GPIO as GPIO

    GPIO.setmode(GPIO.BOARD)

    # Initialize all pins as outputs
    for name, pin in JETSON_GPIO_PINS.items():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

    def set_pin(name: str, state: bool):
        pin = JETSON_GPIO_PINS.get(name)
        if pin is None:
            raise ValueError(f"Unknown GPIO pin: {name}")
        # LOW = on (LED/bzr powered active-low)
        GPIO.output(pin, GPIO.LOW if state else GPIO.HIGH)

    def cleanup():
        GPIO.cleanup()
