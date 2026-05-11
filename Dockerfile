# Use NVIDIA L4T ML base image for Jetson Orin Nano
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_jetson.txt .
RUN pip3 install --no-cache-dir -r requirements_jetson.txt

# Copy source code and UI
COPY src/ ./src/
COPY ui/ ./ui/
COPY data/combined/models/ ./data/combined/models/

# Expose API port
EXPOSE 8000

# Entrypoint script to detect mode
CMD ["python3", "src/api/main.py", "--mode", "camera"]
