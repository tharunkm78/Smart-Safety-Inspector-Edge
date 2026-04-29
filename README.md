# Smart Safety Inspector

AI-powered real-time workplace safety hazard detection system for **Jetson Orin Nano** (JetPack 5.x) with Windows development fallback.

## What It Does

Detects workplace hazards in real-time from camera feeds:
- **PPE violations** — missing hardhat, vest, gloves, goggles, boots, etc.
- **Construction equipment** — excavators, bulldozers, cranes, trucks
- **Situational hazards** — fire, fall risk, vehicle proximity, spills

Alerts workers immediately via audio (buzzer) + visual (LED) feedback and logs all events.

## Quick Start

### Windows (Development)

1. **Setup**: Run `.\scripts\setup_windows.ps1`.
2. **Datasets**: Ensure datasets are in `data/raw/` (PPE and Construction).
3. **Balance**: `python -m src.data.balance_dataset` to prepare the unified dataset.
4. **Train**: `python -m src.training.train_yolov8 --epochs 10` (for a quick test).
5. **Dashboard**: `python -m src.api.main` and open `http://localhost:8000`.

### Jetson Orin Nano (Production)

1. **Setup**: `chmod +x scripts/setup_jetson.sh && ./scripts/setup_jetson.sh`.
2. **Export**: `python -m src.training.export_tensorrt` to create the `.engine` file.
3. **Run**: `./scripts/run_full_system.sh`.

## Architecture

```
Camera → detector.py → alert_manager.py → audio/visual alerts
                          ↓
                     WebSocket → UI Dashboard
                          ↓
                     SQLite (alerts.db)
```

The system uses **FastAPI** on all platforms for the REST API and WebSocket communication. On Windows, hardware alerts are simulated via console output and a UI overlay.

## Key Commands

| Command | Description |
|---|---|
| `python -m src.data.balance_dataset` | Balance and merge datasets |
| `python -m src.data.dataset_stats` | Show class distribution |
| `python -m src.training.train_yolov8` | Train YOLOv8n safety model |
| `python -m src.training.export_tensorrt` | Export to TensorRT engine (for Jetson) |
| `python -m src.inference.detector --camera 0` | Test camera inference |
| `python -m src.api.main` | Start API server + Dashboard |
| `pytest tests/` | Run unit tests |

## Project Structure

```
src/
  config.py           # Single source of truth (classes, thresholds, pins)
  data/               # Dataset balancing and format conversion
  training/           # YOLOv8n fine-tuning + TensorRT export
  inference/          # Detector + camera capture + TensorRT engine
  alerting/           # Audio/visual alerts + SQLite logger
  api/                # FastAPI REST + WebSocket server
ui/                   # Real-time dashboard (HTML/CSS/JS)
models/               # Trained weights registry (.pt and .engine)
```

## Datasets

The system is trained on a combined dataset of **22 safety classes** derived from:
- **PPE Dataset**: `ppe-v2`
- **Construction Equipment**: `construction-equipment`

## Hardware Mapping (Jetson)

| Pin | Function |
|---|---|
| GPIO 7 | Red LED (CRITICAL alert) |
| GPIO 11 | Yellow LED (WARNING alert) |
| GPIO 13 | Green LED (OK status) |
| GPIO 15 | Buzzer trigger |
