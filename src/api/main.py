"""FastAPI main application.

Usage:
    # Jetson / Linux
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

    # Windows dev
    python -m src.api.main
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from src.api.routes import detect, alerts, status
from src.api.websocket_manager import WSManager
from src.api.pipeline import run_pipeline
from src.config import API_CORS_ORIGINS, UI_DIR

app = FastAPI(
    title="Smart Safety Inspector API",
    version="1.0.0",
    description="Real-time safety hazard detection API for wearable AI systems",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_manager = WSManager()
app.state.ws = ws_manager

# Shared Camera instance
_camera = None

def get_camera():
    global _camera
    if _camera is None:
        from src.inference.camera import Camera
        _camera = Camera(camera_idx=0)
    return _camera

@app.on_event("startup")
async def startup_event():
    import asyncio
    print("--- SERVER STARTING ---")
    try:
        cam = get_camera()
        print(f"--- CAMERA INITIALIZED ---")
        asyncio.create_task(run_pipeline(ws_manager, cam))
    except Exception as e:
        print(f"!!! CRITICAL STARTUP ERROR !!!: {e}")

@app.get("/api/video_feed")
async def video_feed():
    """MJPEG streaming endpoint."""
    import cv2
    import time
    cam = get_camera()
    if not cam._running:
        cam.start()

    def generate():
        while True:
            frame = cam.read()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

app.include_router(detect.router, prefix="/api", tags=["Detection"])
app.include_router(alerts.router, prefix="/api", tags=["Alerts"])
app.include_router(status.router, prefix="/api", tags=["Status"])


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    client_host = websocket.client.host
    print(f"--- WS CONNECTING ---: From {client_host}")
    await ws_manager.connect(websocket)
    print(f"--- WS CONNECTED ---: {client_host}")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception:
        pass
    finally:
        ws_manager.disconnect(websocket)

if UI_DIR.exists():
    # Mount assets folders
    if (UI_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(UI_DIR / "css")), name="css")
    if (UI_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(UI_DIR / "js")), name="js")

    @app.get("/")
    async def root():
        ui_index = UI_DIR / "index.html"
        if ui_index.exists():
            return FileResponse(str(ui_index))
        return {"message": "Smart Safety Inspector API", "ui": "missing index.html"}
else:
    @app.get("/")
    async def root():
        return {"message": "Smart Safety Inspector API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
