import os
from pathlib import Path

# -- Project Structure --
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
COMBINED_DATA_DIR = DATA_DIR / "combined"
MODELS_DIR = PROJECT_ROOT / "models"

# -- Unified Safety Classes --
SAFETY_CLASSES = [
    "helmet_on",        # 0
    "gloves_on",        # 1
    "vest_on",          # 2
    "boots",            # 3
    "person",           # 4
    "fire",             # 5
    "smoke",            # 6
]

NUM_CLASSES = len(SAFETY_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(SAFETY_CLASSES)}

# -- Raw Dataset Mappings --
DATASET_MAPPINGS = {
    "ppe": {
        0: "boots",
        2: "gloves_on",
        3: "helmet_on",
        9: "person",
        10: "vest_on",
    },
    "construction-equipment": {
        3: "gloves_on",
        5: "helmet_on",
        8: "vest_on",
        13: "person",
    },
    "Fire-Smoke": {
        0: "fire",
        1: "smoke",
    },
    "person": {
        0: "person",
    }
}

# -- Inference Confidence Thresholds --
MIN_CONF = {
    "person": 0.50,
    "helmet_on": 0.40,
    "vest_on": 0.40,
    "gloves_on": 0.45,
    "boots": 0.45,
    "fire": 0.30,
    "smoke": 0.20,
}

# -- Training Configuration --
TRAIN_CONFIG = {
    "model": "yolov8s.pt", # Switched from nano to small for better multitask performance
    "data": str(COMBINED_DATA_DIR / "dataset.yaml"),
    "epochs": 100,          # Increased for better convergence
    "patience": 20,         # Increased patience
    "imgsz": 768,           # Increased image size for diffuse smoke/fire
    "batch": 16,
    "workers": 4,          # Reduced for Windows stability
    "project": str(MODELS_DIR),
    "name": "major_retrain_v1",
    "exist_ok": True,
    "device": 0,
    
    # Advanced Hyperparameters & Optimizations
    "cos_lr": True,         # Better convergence
    "close_mosaic": 10,     # Stabilize final epochs
    "cache": False,         # Disabled to fix MemoryError on Windows
    "plots": True,          # Essential for debugging
    "freeze": 10,           # Freeze backbone for first stage stability
    
    # Loss Weights
    "box": 8.0,             # Increased box loss for better localization
    "cls": 0.5,
    "dfl": 1.5,
    
    # Advanced Augmentations (Conservative for smoke textures)
    "hsv_h": 0.01,
    "hsv_s": 0.4,
    "hsv_v": 0.2,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.3,
    "shear": 2.0,
    "perspective": 0.0005,
    "fliplr": 0.5,
    "mosaic": 0.7,
    "mixup": 0.05,
}
