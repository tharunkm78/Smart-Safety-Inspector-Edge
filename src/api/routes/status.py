"""GET /api/status — system and model status."""

import time
from pathlib import Path

from fastapi import APIRouter

from src.config import MODEL_PT, MODEL_TRT, is_jetson, SAFETY_CLASSES
from src.alerting.alert_logger import init_db

router = APIRouter()


@router.get("/status")
async def status():
    """Return system and model status."""
    model_path = MODEL_TRT if is_jetson() else MODEL_PT
    model_exists = model_path.exists()

    # DB status
    try:
        init_db()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "platform": "jetson" if is_jetson() else "windows",
        "model_loaded": model_exists,
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 1) if model_exists else 0,
        "num_classes": len(SAFETY_CLASSES),
        "classes": SAFETY_CLASSES,
        "database_ok": db_ok,
        "timestamp": time.time(),
    }


@router.get("/status/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}
