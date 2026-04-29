"""WebSocket manager for real-time UI updates."""

import asyncio
import json
from typing import Optional

from fastapi import WebSocket


class WSManager:
    """Manages WebSocket connections and broadcasts detection/alert events."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, event: dict):
        """Broadcast a JSON event to all connected clients."""
        if not self._connections:
            return
        message = json.dumps(event)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_detections(self, frame_id: str, detections: list[dict], fps: float = 0.0):
        """Broadcast a frame's detection results."""
        await self.broadcast({
            "type": "detections",
            "frame_id": frame_id,
            "detections": detections,
            "fps": round(fps, 1),
        })

    async def broadcast_alert(self, alert: dict):
        """Broadcast a new alert."""
        await self.broadcast({
            "type": "alert",
            "alert": alert,
        })

    async def broadcast_status(self, status: dict):
        """Broadcast system status update."""
        await self.broadcast({
            "type": "status",
            "status": status,
        })
