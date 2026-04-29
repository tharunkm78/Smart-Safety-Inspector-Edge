#!/bin/bash
# Run the full Smart Safety Inspector system on Jetson
# Starts: camera capture → inference → alerting → API → UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Smart Safety Inspector — Full System ==="
echo "Project: $PROJECT_ROOT"

# Check model
if [ ! -f models/yolov8n_safety_v1.engine ]; then
    echo "WARNING: TensorRT engine not found at models/yolov8n_safety_v1.engine"
    echo "  Run: python -m src.training.export_tensorrt"
fi

# Activate venv if exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Start the API + real-time pipeline
echo "Starting API server..."
uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 &

API_PID=$!
echo "API PID: $API_PID"

# Give API time to start
sleep 3

# Start the real-time camera → inference → alerting loop
echo "Starting camera inference pipeline..."
python -c "
import sys
import time
sys.path.insert(0, '$PROJECT_ROOT')

from src.inference.camera import Camera
from src.inference.detector import YOLOv8nDetector
from src.alerting.alert_manager import AlertManager
from src.api.websocket_manager import WSManager
from src.config import is_jetson

print('Loading detector...')
detector = YOLOv8nDetector()
alert_mgr = AlertManager()
ws = WSManager()

print('Starting camera...')
cam = Camera(camera_idx=0)
cam.start()

frame_id = 0
print('Inference loop running. Press Ctrl+C to stop.')
while True:
    frame = cam.read()
    if frame is None:
        continue

    detections = detector.detect(frame)
    alert_mgr.dispatch(detections)

    # Push to WebSocket clients
    import asyncio
    import json

    fps = detector.avg_fps()
    msg = json.dumps({
        'type': 'detections',
        'frame_id': str(frame_id),
        'detections': detections,
        'fps': fps,
    })
    # Note: broadcast is async — in production use threading or queue
    frame_id += 1
    time.sleep(0.03)  # ~30 FPS

    if frame_id % 100 == 0:
        print(f'Frames: {frame_id}, FPS: {fps:.1f}')

except KeyboardInterrupt:
    print('Shutting down...')
    cam.stop()
    alert_mgr.stop()
    print('Done.')
"

# Cleanup on exit
trap "kill $API_PID 2>/dev/null; exit" EXIT
