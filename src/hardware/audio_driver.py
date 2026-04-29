"""Cross-platform audio driver for hardware alerting."""

from src.alerting.audio_alert import AudioAlert

_audio: AudioAlert | None = None


def get_audio_driver() -> AudioAlert:
    global _audio
    if _audio is None:
        _audio = AudioAlert()
    return _audio


def play_alert(priority: str):
    """Play an alert tone for the given priority."""
    get_audio_driver().play(priority)


def beep_sequence(priority: str, count: int = 3):
    """Play a sequence of beeps."""
    get_audio_driver().beep_sequence(priority, count)
