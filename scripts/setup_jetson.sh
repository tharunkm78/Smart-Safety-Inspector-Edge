#!/bin/bash
# Setup script for Jetson Orin Nano (JetPack 5.x)
# Run: chmod +x scripts/setup_jetson.sh && ./scripts/setup_jetson.sh

set -e

echo "=== Smart Safety Inspector — Jetson Setup ==="
echo "Platform: Jetson Orin Nano (JetPack 5.x)"
echo

# Check if running on Jetson
if [ ! -f /proc/device-tree/model ]; then
    echo "ERROR: This script is for Jetson hardware only."
    exit 1
fi

echo "[1/7] Updating package list..."
sudo apt update -y

echo "[2/7] Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopencv-dev \
    libopencv-python \
    v4l-utils \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

echo "[3/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "[4/7] Installing PyTorch for JetPack (L4T)..."
# Download PyTorch wheel from NVIDIA developer site
# Check L4T version
L4T_VERSION=$(head -c 4 /etc/nv_tegra_release)
echo "  L4T version: $L4T_VERSION"

# Download appropriate PyTorch wheel
# For JetPack 5.1 / L4T 35.4:
TORCH_WHL="https://nvidia.box.com/shared/static/h1f1lbf6e7g6evr3vv3si6m1s7s2l2xw.whl"
pip install "$TORCH_WHL" || echo "  PyTorch wheel failed — trying pip install..."

echo "[5/7] Installing other core packages..."
pip install \
    numpy \
    opencv-python==4.9.0.80 \
    pillow \
    requests \
    pyyaml

echo "[6/7] Verifying TensorRT..."
python3 -c "import tensorrt; print(f'  TensorRT {tensorrt.__version__}')" || echo "  TensorRT not detected — install via JetPack"

echo "[7/7] Installing project dependencies..."
pip install -r requirements.txt || pip install -r requirements.jetson.txt

echo
echo "=== Setup Complete ==="
echo "Activate environment: source venv/bin/activate"
echo "Run full system: ./scripts/run_full_system.sh"
echo
echo "Next steps:"
echo "  1. Balance dataset:   python -m src.data.balance_dataset"
echo "  2. Train model:        python -m src.training.train_yolov8"
echo "  3. Export TensorRT:     python -m src.training.export_tensorrt"
echo "  4. Start API:          uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
