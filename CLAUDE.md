# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Smart Safety Inspector** — AI-powered real-time workplace safety hazard detection for Jetson Orin Nano (JetPack 5.x) with Windows development fallback.

- Detects PPE violations (hardhat, vest, gloves, boots, etc.) and construction equipment
- Runs YOLOv8n nano on edge hardware with TensorRT FP16 acceleration
- Alerts workers via audio (buzzer) + visual (LED) with real-time WebSocket dashboard

## Common Commands

### Setup
```bash
# Windows dev (uses FastAPI + Uvicorn)
.\scripts\setup_windows.ps1

# Jetson (on device)
chmod +x scripts/setup_jetson.sh && ./scripts/setup_jetson.sh
```

### Data Pipeline
```bash
# Note: Datasets are pre-downloaded in data/raw/
python -m src.data.balance_dataset      # Merge and oversample to balance classes
python -m src.data.dataset_stats        # Show class distribution report
python -m src.data.convert_annotations # Convert between COCO and YOLO formats
```

### Training
```bash
python -m src.training.train_yolov8 --epochs 100       # Train YOLOv8n (local GPU)
python -m src.training.export_tensorrt                # Export .pt → TensorRT .engine
python -m src.training.validate                        # Evaluate mAP@0.5 / mAP@0.5:0.95
```

### Inference
```bash
python -m src.inference.detector --camera 0            # Live camera detection
python -m src.inference.detector --image test.jpg      # Single image detection
```

### API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000   # Start API + serve UI at /
python -m src.api.main                                 # Start via wrapper (Windows/dev)
pytest tests/                                           # Run unit tests
```

### Full System (Jetson)
```bash
./scripts/run_full_system.sh    # Start camera → inference → alerting → API
```

## Architecture

```
Camera Feed (V4L2 / DirectShow)
  └── src/inference/camera.py        Camera capture (threaded, non-blocking)
       └── src/inference/detector.py     YOLOv8nDetector (PyTorch or TensorRT)
            └── src/inference/postprocess.py  NMS, confidence filtering
                 └── src/alerting/alert_manager.py   Priority dispatch
                          ├── src/alerting/audio_alert.py   Buzzer (winsound / pygame)
                          ├── src/alerting/visual_alert.py  LED patterns (GPIO / tkinter)
                          ├── src/alerting/alert_logger.py   SQLite alert history
                          └── src/api/websocket_manager.py   Real-time push to UI

src/api/main.py  ── FastAPI REST + WebSocket server
ui/              ── Dashboard (HTML/CSS/JS + Chart.js, WebSocket client)
```

## Development Guidelines

1. **Imports**: All scripts use `sys.path.insert(0, ...)` to point to the project root. Always use absolute package imports: `from src.config import ...`.
2. **Configuration**: All settings are in `src/config.py`. No hardcoded values (classes, thresholds, GPIO pins) in other files.
3. **Platform Detection**: Use `src.config.is_windows()` and `is_jetson()` for cross-platform logic.
4. **Hardware**: GPIO pins (7, 11, 13, 15) are mapped to Red/Yellow/Green LEDs and Buzzer on Jetson. Windows fallbacks print to console.
5. **Data**: Unified class list contains 22 safety-related classes (PPE + Construction).

## Alert Priority System

| Priority | Trigger | Response |
|---|---|---|
| CRITICAL | fire, fall_risk, vehicle_proximity | Audio beep × 5 + rapid LED blink + log |
| HIGH | no_ppe, spill | Audio beep × 3 + yellow LED + log |
| MEDIUM | Missing individual PPE item | Yellow LED slow blink + log |
| LOW | Equipment/person (no hazard) | Log only |

Cooldowns prevent alert spam: CRITICAL=3s, HIGH=5s, MEDIUM=10s, LOW=15s.
