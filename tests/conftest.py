"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerting.alert_logger import init_db
from src.config import ALERT_DB_PATH


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Provide a fresh temporary database for each test."""
    import shutil
    # Override the global db path for tests
    import src.alerting.alert_logger as al
    test_db = tmp_path / "test_alerts.db"
    al.ALERT_DB_PATH = test_db
    init_db(test_db)
    yield test_db
    # Restore
    al.ALERT_DB_PATH = ALERT_DB_PATH
