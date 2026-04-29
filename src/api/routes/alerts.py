"""GET /api/alerts — alert history and acknowledgment."""

from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.alerting.alert_logger import get_alerts, acknowledge_alert, alert_stats

router = APIRouter()


class AlertResponse(BaseModel):
    id: int
    timestamp: str
    class_name: str
    class_id: int
    confidence: float
    priority: str
    bbox: list
    acknowledged: bool


@router.get("/alerts")
async def list_alerts(
    limit: int = Query(default=100, le=1000),
    priority: Optional[str] = None,
    acknowledged: Optional[bool] = None,
):
    """Get alert history with optional filters."""
    alerts = get_alerts(limit=limit, priority=priority, acknowledged=acknowledged)
    return {"alerts": alerts, "count": len(alerts)}


@router.post("/alerts/{alert_id}/ack")
async def ack_alert(alert_id: int):
    """Acknowledge an alert."""
    acknowledge_alert(alert_id)
    return {"status": "ok", "alert_id": alert_id}


@router.get("/alerts/stats")
async def stats():
    """Get alert statistics."""
    return alert_stats()
