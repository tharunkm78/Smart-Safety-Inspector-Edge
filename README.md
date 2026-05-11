# 🛡️ Smart Safety Inspector: Tactical Edge HUD
**Industrial-Grade Workplace Safety & Hazard Monitoring System**

![Dashboard Preview](https://img.shields.io/badge/UI-Tactical_HUD-red)
![Target](https://img.shields.io/badge/Hardware-Jetson_Orin_Nano-green)
![Logic](https://img.shields.io/badge/Safety_Logic-3--Tier_Risk-orange)

The **Smart Safety Inspector** is a high-performance computer vision solution designed for real-time safety compliance monitoring. Featuring a state-of-the-art "Tactical HUD" interface, the system detects PPE violations, fire hazards, and unauthorized personnel in industrial environments.

---

## 🚀 Core Features
*   **Tactical Surveillance HUD**: A high-engagement, dark-mode dashboard providing real-time visual overlays and threat assessments.
*   **3-Tier Safety Logic**:
    *   🟢 **SAFE**: Full PPE compliance.
    *   🟡 **WARNING**: Minor violations (e.g., missing gloves).
    *   🔴 **CRITICAL**: Major hazards (Fire, Smoke, or Multiple PPE failures).
*   **Boundary-Aware Intelligence**: Smart detection logic that ignores "missing boots" if a worker's feet are off-camera.
*   **Platform Agnostic**: Optimized for both **Windows Development** (Simulation Mode) and **NVIDIA Jetson Orin Nano** (Live Production).
*   **Zero-Latency Normalization**: Uses normalized coordinate systems to ensure perfect UI rendering across any camera resolution.

---

## 🛠️ Technical Stack
*   **Inference**: YOLOv8 (Ultralytics)
*   **Backend**: FastAPI / Python 3.10+
*   **Frontend**: Vanilla JS / Canvas API (High Performance)
*   **Streaming**: WebSocket-based telemetry & MJPEG Video Feed
*   **Deployment**: Docker (Jetson L4T optimized)

---

## 📦 Quick Start

### 1. Development (Simulation Mode)
Run the inspector using the provided test dataset to verify logic and UI.
```powershell
python src/api/main.py --mode test
```

### 2. Production (Live Camera)
Run with a live camera feed (USB/CSI).
```bash
python3 src/api/main.py --mode camera --source 0
```

---

## 🐳 Docker Deployment (Jetson)
Ensure hardware acceleration on Jetson hardware.
```bash
chmod +x run_jetson.sh
./run_jetson.sh camera
```

---

## 📁 Repository Structure
*   `src/api/`: FastAPI server and WebSocket logic.
*   `src/inference/`: YOLO detection and 3-Tier Safety Logic Engine.
*   `ui/`: Dashboard assets (HTML/CSS/JS).
*   `Dockerfile`: Jetson-optimized production container.

---
**Developed by tharunkm78**  
*Hardening Workplace Safety at the Edge.*
