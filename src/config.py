"""Smart Safety Inspector — Configuration

Single source of truth for all paths, class names, thresholds, hardware pins, and API keys.
Built from the actual downloaded datasets: ppe-v2 and construction-equipment.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
UI_DIR = PROJECT_ROOT / "ui"

# ─── Dataset Paths ───────────────────────────────────────────────────────────
RAW_DATA_DIR = DATA_DIR / "raw"
PPE_DIR = RAW_DATA_DIR / "ppe-v2"
CONSTRUCTION_DIR = RAW_DATA_DIR / "construction-equipment"

# ─── Unified Safety Classes ──────────────────────────────────────────────────
# PPE dataset (ppe-v2): 8 classes, IDs 0-7
# Construction dataset: 14 classes, IDs 0-13
# Unified list: 22 classes (0-21), with deduplication of "Vest" variants
#
# Unified class list (22 total):
#   0-7:   PPE items (from ppe-v2)
#   8-21:  Construction equipment + context (from construction-equipment)

SAFETY_CLASSES = [
    # PPE items (from ppe-v2, IDs 0-7)
    "dust_mask",        # 0  — Dust mask
    "eye_wear",         # 1  — Eye Wear
    "glove",            # 2  — Glove (PPE)
    "jacket",           # 3  — Jacket
    "boots",            # 4  — Protective Boots
    "helmet",           # 5  — Protective Helmet / Hard Hat
    "shield",           # 6  — Shield
    "person",           # 7  — people
    # Construction equipment (IDs 8-21)
    "dump_truck",       # 8  — Dump Truck
    "excavator",        # 9  — Excavator (NEW)
    "front_end_loader", # 10 — Front End Loader
    "gloves",           # 11 — Gloves (construction context — distinct from PPE glove)
    "hardhat_off",      # 12 — Hard Hat OFF (safety violation)
    "hardhat_on",       # 13 — Hard Hat ON
    "ladder",           # 14 — Ladder (NEW)
    "vest_off",         # 15 — Safety Vest OFF (safety violation)
    "vest_on",          # 16 — Safety Vest ON
    "skid_steer",       # 17 — Skid Steer
    "tractor_trailer",  # 18 — Tractor Trailer
    "trailer",          # 19 — Trailer (NEW)
    "vehicle",          # 20 — Vehicle (construction)
    "worker",           # 21 — Worker (NEW — construction context)
]

NUM_CLASSES = len(SAFETY_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(SAFETY_CLASSES)}

# ─── Raw Dataset Class Names ─────────────────────────────────────────────────
# PPE dataset (ppe-v2/data.yaml) — 8 classes
PPE_CLASS_NAMES = [
    "Dust mask", "Eye Wear", "Glove", "Jacket",
    "Protective Boots", "Protective Helmet", "Shield", "people",
]

# Construction dataset (construction-equipment/data.yaml) — 14 classes
CONSTRUCTION_CLASS_NAMES = [
    "Dump Truck", "Excavator", "Front End Loader", "Gloves",  # 0-3
    "Hard Hat OFF", "Hard Hat ON", "Ladder",                     # 4-6
    "Safety Vest OFF", "Safety Vest ON", "Skid Steer",            # 7-9
    "Tractor Trailer", "Trailer", "Vehicle", "Worker",           # 10-13
]

# ─── Alert Priority Mapping ─────────────────────────────────────────────────
ALERT_PRIORITY = {
    # Safety violations — HIGH
    "hardhat_off": "HIGH",
    "vest_off": "HIGH",
    # PPE items present — MEDIUM
    "helmet": "MEDIUM",
    "hardhat_on": "MEDIUM",
    "vest_on": "MEDIUM",
    "dust_mask": "MEDIUM",
    "eye_wear": "MEDIUM",
    "glove": "MEDIUM",
    "jacket": "MEDIUM",
    "boots": "MEDIUM",
    "shield": "MEDIUM",
    # Context — LOW
    "person": "LOW",
    "worker": "LOW",
    # Equipment — LOW
    "dump_truck": "LOW",
    "excavator": "LOW",
    "front_end_loader": "LOW",
    "skid_steer": "LOW",
    "tractor_trailer": "LOW",
    "trailer": "LOW",
    "vehicle": "LOW",
    "ladder": "LOW",
    "gloves": "LOW",      # construction gloves (different context from PPE glove)
}

# ─── Inference Thresholds ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.45

# ─── Alert Cooldowns (seconds) ──────────────────────────────────────────────
ALERT_COOLDOWNS = {
    "CRITICAL": 3,
    "HIGH": 5,
    "MEDIUM": 10,
    "LOW": 15,
}

# ─── Model Config ───────────────────────────────────────────────────────────
MODEL_NAME = "yolov8n_safety_v1"
MODEL_PT = MODELS_DIR / f"{MODEL_NAME}.pt"
MODEL_TRT = MODELS_DIR / f"{MODEL_NAME}.engine"
MODEL_TORCHSCRIPT = MODELS_DIR / f"{MODEL_NAME}.torchscript"

TRAIN_CONFIG = {
    "model": "yolov8n.pt",
    "data": str(DATA_DIR / "combined" / "dataset.yaml"),
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "patience": 15,
    "device": None, # Set to None for auto-detection in train_yolov8.py
    "augment": True,
    "mosaic": 1.0,
    "copy_paste": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "cls": 1.5,
    "workers": 0,
    "project": str(MODELS_DIR),
    "name": "train_run",
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
}

# ─── Hardware Pin Mapping (Jetson Orin Nano GPIO) ───────────────────────────
JETSON_GPIO_PINS = {
    "LED_RED": 7,
    "LED_YELLOW": 11,
    "LED_GREEN": 13,
    "BUZZER": 15,
}

# ─── Platform Detection ─────────────────────────────────────────────────────
def is_jetson() -> bool:
    return os.path.exists("/proc/device-tree/model")

def is_windows() -> bool:
    return os.name == "nt"

PLATFORM = "jetson" if is_jetson() else ("windows" if is_windows() else "linux")

# ─── Camera Config ──────────────────────────────────────────────────────────
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "backend_windows": "DirectShow",
    "backend_linux": "V4L2",
}

# ─── API Config ─────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_CORS_ORIGINS = ["*"]

# ─── Alert Database ─────────────────────────────────────────────────────────
ALERT_DB_PATH = LOGS_DIR / "alerts.db"

# ─── Balance Config ─────────────────────────────────────────────────────────
MAX_CLASS_RATIO = 8.0           # max ratio between most/least frequent class (after oversampling)
MAX_SAMPLES_PER_CLASS = 4000   # absolute cap to avoid excessive oversampling
MIN_SAMPLES_PER_CLASS = 50     # floor for rare classes
