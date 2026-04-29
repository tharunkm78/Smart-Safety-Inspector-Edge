"""Simulated GPIO for Windows development.

No actual hardware — prints to console and updates tkinter indicator.
This allows all code to run and be tested on Windows before deployment.
"""


def set_pin(name: str, state: bool):
    """Simulate setting a GPIO pin on Windows."""
    led_state = "ON" if state else "OFF"
    pin_map = {
        "LED_RED": "🔴",
        "LED_YELLOW": "🟡",
        "LED_GREEN": "🟢",
        "BUZZER": "🔔",
    }
    icon = pin_map.get(name, "❓")
    print(f"[GPIO Windows] {name} → {led_state} {icon}")


def cleanup():
    """No-op on Windows."""
    pass
