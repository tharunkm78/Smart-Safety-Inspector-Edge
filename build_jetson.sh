#!/bin/bash
# build_jetson.sh - Dedicated Build Script

echo "--- Smart Safety Inspector: Building Production Container ---"
docker build -t safety-inspector .
echo "Build complete! You can now use ./run_jetson.sh to launch instantly."
