import asyncio
import base64
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import CAMERA_CONFIG, MODEL_PT, MODEL_TRT, is_jetson
from src.inference.camera import Camera
from src.inference.detector import YOLOv8nDetector
from src.alerting.alert_manager import AlertManager


async def run_pipeline(ws_manager, cam):
    """Background task to run camera inference and broadcast to WebSocket."""
    print("Starting background inference pipeline...")
    
    # Initialize components
    model_path = MODEL_TRT if is_jetson() else MODEL_PT
    if not model_path.exists():
        print(f"!!! MODEL ERROR !!!: No model found at {model_path.absolute()}")
        print(f"Looking for: {MODEL_PT.name}")
        detector = None
    else:
        print(f"--- MODEL DETECTED ---: {model_path.name}")
        try:
            detector = YOLOv8nDetector(model_path=model_path)
            print(f"--- MODEL LOADED SUCCESSFULLY ---")
        except Exception as e:
            print(f"!!! MODEL LOAD ERROR !!!: {e}")
            detector = None
    
    alert_mgr = AlertManager()
    
    try:
        if not cam._running:
            cam.start()
        frame_id = 0
        
        while True:
            # Check for connected clients to avoid wasting CPU
            if not ws_manager._connections:
                await asyncio.sleep(1.0)
                continue
                
            frame = cam.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            # Run detection
            detections = []
            fps = 0.0
            if detector:
                detections = detector.detect(frame)
                alert_mgr.dispatch(detections)
                fps = detector.avg_fps()
            
            # Broadcast
            if frame_id % 100 == 0:
                print(f"Broadcasting frame {frame_id}, {len(detections)} detections")
            
            await ws_manager.broadcast({
                "type": "detections",
                "frame_id": str(frame_id),
                "detections": detections,
                "fps": fps
            })
            
            frame_id += 1
            # Control loop rate (e.g., 20 FPS)
            await asyncio.sleep(0.05)
            
    except asyncio.CancelledError:
        print("Pipeline task cancelled.")
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping camera...")
        cam.stop()
        alert_mgr.stop()
