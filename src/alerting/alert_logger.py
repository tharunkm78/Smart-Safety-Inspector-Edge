"""Alert logger — SQLite-based alert history.

Schema:
    alerts(id, timestamp, class_name, class_id, confidence, priority, bbox, acknowledged)
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import ALERT_DB_PATH


def init_db(db_path: Optional[Path] = None):
    """Initialize the alerts database."""
    db_path = db_path or ALERT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            priority TEXT NOT NULL,
            bbox TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON alerts(priority)")
    conn.commit()
    return conn


def log_alert(
    class_name: str,
    class_id: int,
    confidence: float,
    priority: str,
    bbox: list,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a new alert into the database. Returns the row ID."""
    db_path = db_path or ALERT_DB_PATH
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """
        INSERT INTO alerts (timestamp, class_name, class_id, confidence, priority, bbox)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.now(timezone.utc).isoformat(), class_name, class_id, confidence, priority, json.dumps(bbox)),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_alerts(
    limit: int = 100,
    priority: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """Retrieve alert history."""
    db_path = db_path or ALERT_DB_PATH
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    query = "SELECT id, timestamp, class_name, class_id, confidence, priority, bbox, acknowledged FROM alerts"
    conditions = []
    params = []

    if priority:
        conditions.append("priority = ?")
        params.append(priority)
    if acknowledged is not None:
        conditions.append("acknowledged = ?")
        params.append(1 if acknowledged else 0)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "class_name": r[2],
            "class_id": r[3],
            "confidence": r[4],
            "priority": r[5],
            "bbox": json.loads(r[6]),
            "acknowledged": bool(r[7]),
        }
        for r in rows
    ]


def acknowledge_alert(alert_id: int, db_path: Optional[Path] = None):
    """Mark an alert as acknowledged."""
    db_path = db_path or ALERT_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def alert_stats(db_path: Optional[Path] = None) -> dict:
    """Return alert statistics."""
    db_path = db_path or ALERT_DB_PATH
    if not db_path.exists():
        return {"total": 0, "by_priority": {}, "by_class": {}}

    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    by_priority = dict(conn.execute(
        "SELECT priority, COUNT(*) FROM alerts GROUP BY priority"
    ).fetchall())
    by_class = dict(conn.execute(
        "SELECT class_name, COUNT(*) FROM alerts GROUP BY class_name"
    ).fetchall())
    conn.close()
    return {"total": total, "by_priority": by_priority, "by_class": by_class}
