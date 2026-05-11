import os
import random
from ultralytics import YOLO

import sys
from pathlib import Path

# Add src to path to import config (Runtime fix)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from src.config (IDE-friendly import)
from src.config import MODELS_DIR, COMBINED_DATA_DIR, TRAIN_CONFIG


# Define the paths
test_dir = COMBINED_DATA_DIR / "images" / "test"
model_path = MODELS_DIR / TRAIN_CONFIG["name"] / "weights" / "best.pt"

if not model_path.exists():
    print(f"Warning: Trained model not found at {model_path}")
    print(f"Falling back to base model: {TRAIN_CONFIG['model']}")
    model_path = TRAIN_CONFIG["model"]

# Load the trained model
print(f"Loading model from: {model_path}")
model = YOLO(str(model_path))

# Grab all images in the test directory and pick 5 randomly
all_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
selected_images = random.sample(all_images, min(5, len(all_images)))

print(f"Testing on {len(selected_images)} random images...\n")

# Run Inference
# save=True saves the output with bounding boxes to a 'runs/detect/' folder
# show=True will open windows to display the images live
results = model.predict(source=selected_images, save=True, show=True)

print("\nDone! Check the 'runs/detect' directory for the saved images with detection boxes.")
