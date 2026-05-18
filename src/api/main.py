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

# Global state mapping camera_index to its latest data
global_camera_state = {}

# Parse CLI Arguments
parser = argparse.ArgumentParser(description="Smart Safety Inspector API")
parser.add_argument("--mode", type=str, default="test", choices=["test", "camera"], help="Run mode: test or camera")
parser.add_argument("--sources", type=str, default="0", help="Comma-separated camera indices")
args, unknown = parser.parse_known_args()

def inference_worker(camera_idx, source_val):
    global global_camera_state
    
    # Initialize the detector for this specific source
    if args.mode == "camera":
        detector = YoloDetector(mode="camera", source=int(source_val))
    else:
        test_images_path = COMBINED_DATA_DIR / "images" / "test"
        detector = YoloDetector(mode="test", source=str(test_images_path))
        
    last_update_time = 0
    last_time = time.time()
    frames_count = 0
    current_frame = None
    
    # Initialize state
    global_camera_state[camera_idx] = {
        "detections": [],
        "fps": 0.0,
        "width": 640,
        "height": 480,
        "frame_id": 0,
        "image": None
    }
    
    while True:
        # Update image based on mode
        if args.mode == "test":
            # Rotate test images every 2 seconds for a slideshow effect
            if time.time() - last_update_time >= 2 or current_frame is None:
                current_frame, current_detections = detector.get_frame_and_detections()
                if current_frame is not None:
                    h, w = current_frame.shape[:2]
                    global_camera_state[camera_idx]["height"] = h
                    global_camera_state[camera_idx]["width"] = w
                    global_camera_state[camera_idx]["frame_id"] += 1
                last_update_time = time.time()
            else:
                time.sleep(0.1)
                continue
        else:
            # Real-time Camera Mode
            current_frame, current_detections = detector.get_frame_and_detections()
            if current_frame is not None:
                h, w = current_frame.shape[:2]
                global_camera_state[camera_idx]["height"] = h
                global_camera_state[camera_idx]["width"] = w
                global_camera_state[camera_idx]["frame_id"] += 1

        if current_frame is None:
            time.sleep(0.1)
            continue
            
        global_camera_state[camera_idx]["detections"] = current_detections
        
        # Calculate FPS (simulated for static stream)
        frames_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            global_camera_state[camera_idx]["fps"] = frames_count / (current_time - last_time)
            frames_count = 0
            last_time = current_time

        # Encode frame as Base64 JPEG for WebSocket
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if ret:
            import base64
            frame_bytes = buffer.tobytes()
            global_camera_state[camera_idx]["image"] = base64.b64encode(frame_bytes).decode('utf-8')
        
        # Small sleep to prevent 100% CPU usage
        time.sleep(0.05)

@app.on_event("startup")
def startup_event():
    import threading
    sources = args.sources.split(",")
    for idx, source_val in enumerate(sources):
        thread = threading.Thread(target=inference_worker, args=(idx, source_val), daemon=True)
        thread.start()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Package all cameras into an array
            feeds_data = []
            for cam_idx, state in global_camera_state.items():
                feeds_data.append({
                    "camera_index": cam_idx,
                    "detections": state["detections"],
                    "fps": state["fps"],
                    "width": state["width"],
                    "height": state["height"],
                    "frame_id": state["frame_id"],
                    "image": state["image"]
                })
                
            data = {
                "type": "multi_camera",
                "feeds": feeds_data
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
