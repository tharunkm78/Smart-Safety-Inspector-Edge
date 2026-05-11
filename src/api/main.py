import asyncio
import cv2
import json
import time
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.detector import YoloDetector
from config import PROJECT_ROOT, COMBINED_DATA_DIR

app = FastAPI()

# Global state for sharing the latest detections between threads/tasks
latest_detections = []
latest_fps = 0.0
current_width = 640
current_height = 480
latest_frame_id = 0

# Parse CLI Arguments
parser = argparse.ArgumentParser(description="Smart Safety Inspector API")
parser.add_argument("--mode", type=str, default="test", choices=["test", "camera"], help="Run mode: test or camera")
parser.add_argument("--source", type=int, default=0, help="Camera index for camera mode")
args = parser.parse_args()

# Initialize the detector based on mode
if args.mode == "camera":
    detector = YoloDetector(mode="camera", source=args.source)
else:
    test_images_path = COMBINED_DATA_DIR / "images" / "test"
    detector = YoloDetector(mode="test", source=str(test_images_path))

def generate_frames():
    global latest_detections, latest_fps, current_width, current_height, latest_frame_id
    
    last_update_time = 0
    last_time = time.time()
    frames_count = 0
    current_frame = None
    current_detections = []
    
    while True:
        # Update image based on mode
        if args.mode == "test":
            if time.time() - last_update_time >= 10 or current_frame is None:
                current_frame, current_detections = detector.get_frame_and_detections()
                if current_frame is not None:
                    current_height, current_width = current_frame.shape[:2]
                    latest_frame_id += 1
                last_update_time = time.time()
        else:
            # Real-time Camera Mode
            current_frame, current_detections = detector.get_frame_and_detections()
            if current_frame is not None:
                current_height, current_width = current_frame.shape[:2]
                latest_frame_id += 1

        if current_frame is None:
            time.sleep(0.1)
            continue
            
        latest_detections = current_detections
        
        # Calculate FPS (simulated for static stream)
        frames_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            latest_fps = frames_count / (current_time - last_time)
            frames_count = 0
            last_time = current_time

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small sleep to prevent 100% CPU usage on the stream loop
        time.sleep(0.05)

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send detections at 10Hz (100ms) to balance real-time tracking with UI stability
            data = {
                "type": "detections",
                "detections": latest_detections,
                "fps": latest_fps,
                "width": current_width,
                "height": current_height,
                "frame_id": latest_frame_id
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected")

# Serve the UI static files
ui_path = PROJECT_ROOT / "ui"
app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Safety Inspector API Server...")
    print("Dashboard available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
