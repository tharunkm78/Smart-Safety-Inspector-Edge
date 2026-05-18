#!/bin/bash
# run_jetson.sh - Production Launch Script (Universal)

MODE=${1:-camera}
echo "--- Smart Safety Inspector: Jetson Deployment ($MODE mode) ---"

DEVICE_ARGS=""
SOURCES=""
for i in 0 1 2 3; do
    if [ -e /dev/video$i ]; then
        DEVICE_ARGS="$DEVICE_ARGS --device /dev/video$i:/dev/video$i"
        if [ -z "$SOURCES" ]; then
            SOURCES="$i"
        else
            SOURCES="$SOURCES,$i"
        fi
    fi
done

if [ -z "$SOURCES" ]; then
    SOURCES="0"
fi

# 2. Run the container
echo "Launching Dashboard with devices: $SOURCES..."
docker run --runtime nvidia -it --rm \
    $DEVICE_ARGS \
    --network host \
    -p 8000:8000 \
    safety-inspector \
    python3 src/api/main.py --mode $MODE --sources $SOURCES
