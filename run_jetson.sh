#!/bin/bash
# run_jetson.sh - Production Launch Script (Universal)

MODE=${1:-camera}
echo "--- Smart Safety Inspector: Jetson Deployment ($MODE mode) ---"

# 1. Build the Docker image
echo "Building Production Container..."
docker build -t safety-inspector .

# 2. Run the container
echo "Launching Dashboard..."
docker run --runtime nvidia -it --rm \
    --device /dev/video0:/dev/video0 \
    --network host \
    -p 8000:8000 \
    safety-inspector \
    python3 src/api/main.py --mode $MODE
