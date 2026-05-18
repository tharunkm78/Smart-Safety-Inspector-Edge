# Use NVIDIA L4T ML base image for Jetson Orin Nano
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set cargo network fetch as a fallback
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
RUN pip3 install --upgrade pip

# Copy requirements and install
COPY requirements_jetson.txt .
RUN pip3 install --no-cache-dir -r requirements_jetson.txt

# Manually install Ultralytics dependencies (skipping opencv and torch since Jetson has them)
RUN pip3 install --no-cache-dir pandas pyyaml tqdm matplotlib seaborn psutil thop

# Install Ultralytics without dependencies to PREVENT it from downloading a broken OpenCV wheel
RUN pip3 install --no-cache-dir --no-deps ultralytics

# Copy source code and UI
COPY src/ ./src/
COPY ui/ ./ui/
COPY models/ ./models/
COPY data/combined/images/test/ ./data/combined/images/test/

# Expose API port
EXPOSE 8000

# Entrypoint script to detect mode
CMD ["python3", "src/api/main.py", "--mode", "camera"]
