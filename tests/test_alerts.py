"""Tests for the alert logger and alert manager."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerting.alert_logger import log_alert, get_alerts, acknowledge_alert, alert_stats
from src.alerting.alert_manager import AlertManager
from src.config import ALERT_DB_PATH


def test_log_alert(fresh_db):
    row_id = log_alert(
        class_name="helmet",
        class_id=5,
        confidence=0.87,
        priority="MEDIUM",
        bbox=[10, 20, 100, 120],
        db_path=fresh_db,
    )
    assert row_id == 1

    alerts = get_alerts(db_path=fresh_db)
    assert len(alerts) == 1
    assert alerts[0]["class_name"] == "helmet"
    assert alerts[0]["priority"] == "MEDIUM"
    assert alerts[0]["acknowledged"] is False


def test_acknowledge_alert(fresh_db):
    row_id = log_alert("hardhat_off", 10, 0.95, "HIGH", [0, 0, 100, 100], db_path=fresh_db)
    assert row_id == 1

    acknowledge_alert(row_id, db_path=fresh_db)
    alerts = get_alerts(db_path=fresh_db)
    assert alerts[0]["acknowledged"] is True


def test_get_alerts_by_priority(fresh_db):
    log_alert("hardhat_off", 10, 0.95, "HIGH", [0, 0, 1, 1], db_path=fresh_db)
    log_alert("helmet", 5, 0.87, "MEDIUM", [0, 0, 1, 1], db_path=fresh_db)
    log_alert("vest_on", 13, 0.75, "MEDIUM", [0, 0, 1, 1], db_path=fresh_db)

    high = get_alerts(priority="HIGH", db_path=fresh_db)
    assert len(high) == 1

    medium = get_alerts(priority="MEDIUM", db_path=fresh_db)
    assert len(medium) == 2


def test_alert_stats(fresh_db):
    log_alert("hardhat_off", 10, 0.95, "HIGH", [0, 0, 1, 1], db_path=fresh_db)
    log_alert("helmet", 5, 0.87, "MEDIUM", [0, 0, 1, 1], db_path=fresh_db)
    log_alert("vest_on", 13, 0.75, "MEDIUM", [0, 0, 1, 1], db_path=fresh_db)

    stats = alert_stats(db_path=fresh_db)
    assert stats["total"] == 3
    assert stats["by_priority"]["HIGH"] == 1
    assert stats["by_priority"]["MEDIUM"] == 2


def test_alert_manager_dispatch(monkeypatch):
    """Test that AlertManager dispatches correctly."""
    dispatched = []

    class MockAudio:
        def beep_sequence(self, *args): dispatched.append(("audio", args))
        def play(self, *args): dispatched.append(("audio", args))

    class MockVisual:
        def trigger(self, p): dispatched.append(("visual", p))
        def set_status(self, p): dispatched.append(("status", p))

    mgr = AlertManager.__new__(AlertManager)
    mgr._audio = MockAudio()
    mgr._visual = MockVisual()
    mgr._last_alert = {}
    mgr._lock = __import__("threading").Lock()
    mgr._active = True

    detections = [
        {"class": "hardhat_off", "class_id": 10, "confidence": 0.95, "priority": "HIGH", "bbox": [0, 0, 100, 100]},
    ]

    mgr.dispatch(detections)

    # HIGH priority triggers audio beep + visual
    assert any("audio" in d[0] for d in dispatched)
    assert any("visual" in d[0] for d in dispatched)


def test_alert_manager_cooldown(monkeypatch):
    """Test that cooldown prevents duplicate alerts within the cooldown window."""
    call_count = []

    class MockAudio:
        def beep_sequence(self, *args): call_count.append(1)
        def play(self, *args): call_count.append(1)

    class MockVisual:
        def trigger(self, p): pass
        def set_status(self, p): pass

    mgr = AlertManager.__new__(AlertManager)
    mgr._audio = MockAudio()
    mgr._visual = MockVisual()
    mgr._last_alert = {}
    mgr._lock = __import__("threading").Lock()
    mgr._active = True

    detections = [
        {"class": "hardhat_off", "class_id": 10, "confidence": 0.95, "priority": "HIGH", "bbox": [0, 0, 100, 100]},
    ]

    # First dispatch
    mgr.dispatch(detections)
    # Second dispatch immediately (within cooldown)
    mgr.dispatch(detections)

    # Only one audio call due to cooldown
    assert len(call_count) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
