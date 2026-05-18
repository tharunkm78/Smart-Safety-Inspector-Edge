import os
import random
import copy
from pathlib import Path

# --- JETSON HEADLESS OPENCV PATCH ---
import cv2
if not hasattr(cv2, 'imshow'):
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: None
    cv2.destroyAllWindows = lambda *args, **kwargs: None
# ------------------------------------

from ultralytics import YOLO
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, TRAIN_CONFIG, SAFETY_CLASSES, MIN_CONF, COMBINED_DATA_DIR

class SafetyLogicEngine:
    """Processes raw detections to derive complex safety violations."""
    
    @staticmethod
    def associate_and_analyze(detections_input):
        # Deep copy to prevent in-place modifications that cause state drift/flickering
        detections = copy.deepcopy(detections_input)
        persons = [d for d in detections if d["class"] == "person"]
        ppe_items = [d for d in detections if d["class"] in ["helmet_on", "vest_on", "gloves_on", "boots"]]
        hazards = [d for d in detections if d["class"] in ["fire", "smoke"]]
        
        results = []
        
        # 1. Process Hazards
        for h in hazards:
            # Smoke and Fire are both critical in wildfire detection
            h["priority"] = "CRITICAL"
            results.append(h)

        # 2. Process Persons and their PPE
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            p_w = px2 - px1
            p_h = py2 - py1
            
            # Define refined regions
            head_zone = (py1, py1 + 0.20 * p_h)
            torso_zone = (py1 + 0.25 * p_h, py1 + 0.75 * p_h)
            # Refined glove zone: broader to catch extended or raised arms
            glove_zone_y = (py1 + 0.15 * p_h, py1 + 0.90 * p_h)
            feet_zone = (py1 + 0.80 * p_h, py2)
            
            # Check for frame truncation (normalized coords 0-1)
            # If a person is cut off at the bottom, we cannot reliably see boots.
            # If cut off at the top, we cannot reliably see a helmet.
            is_truncated_bottom = py2 > 0.95
            is_truncated_top    = py1 < 0.05
            
            associated = {
                "helmet": True if is_truncated_top else False,
                "vest":   False,
                "gloves": False,
                "boots":  True if is_truncated_bottom else False
            }
            
            # Note: We still attempt to find them if they ARE visible even in partial shots
            for ppe in ppe_items:
                ppx1, ppy1, ppx2, ppy2 = ppe["bbox"]
                cx, cy = (ppx1 + ppx2) / 2, (ppy1 + ppy2) / 2
                
                # Check if center is inside person bbox horizontally (with 10% arm buffer)
                if px1 - 0.10 * p_w <= cx <= px2 + 0.10 * p_w:
                    if head_zone[0] <= cy <= head_zone[1] and ppe["class"] == "helmet_on":
                        associated["helmet"] = True
                    elif torso_zone[0] <= cy <= torso_zone[1] and ppe["class"] == "vest_on":
                        associated["vest"] = True
                    elif glove_zone_y[0] <= cy <= glove_zone_y[1] and ppe["class"] == "gloves_on":
                        associated["gloves"] = True
                    elif feet_zone[0] <= cy <= feet_zone[1] and ppe["class"] == "boots":
                        associated["boots"] = True
            
            # Count missing items
            missing_count = list(associated.values()).count(False)
            
            # Apply 3-Tier Rules
            if not associated["helmet"] and not associated["vest"]:
                p["priority"] = "CRITICAL"
                p["class"] = "CRITICAL VIOLATION"
            elif missing_count >= 3:
                p["priority"] = "CRITICAL"
                p["class"] = "MULTIPLE PPE FAILURE"
            elif missing_count > 0:
                p["priority"] = "MEDIUM"
                p["class"] = "PPE WARNING"
            else:
                p["priority"] = "LOW"
                p["class"] = "SAFE WORKER"
                
            # Update label with specifics
            missing_labels = [k for k, v in associated.items() if not v]
            if missing_labels:
                p["class"] += f": Missing {', '.join(missing_labels)}"
                
            results.append(p)

        # 3. Add unassociated PPE for visualization
        for ppe in ppe_items:
            ppe["priority"] = "LOW"
            results.append(ppe)
            
        return results

class YoloDetector:
    def __init__(self, mode='test', source=0):
        self.mode = mode
        self.is_camera = (mode == 'camera')
        
        # Temporal persistence placeholder
        self.violation_memory = {}
        
        model_path = MODELS_DIR / TRAIN_CONFIG["name"] / "weights" / "best.pt"
        if model_path.exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(TRAIN_CONFIG["model"])
            
        self.source = source
        self.cap = None
        self.image_files = []
        
        if self.is_camera:
            import cv2
            print(f"Initializing Camera Source: {source}")
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"ERROR: Could not open camera {source}")
        else:
            # Simulation mode
            source_path = Path(source)
            if source_path.is_dir():
                self.image_files = [
                    f for f in source_path.iterdir() 
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
                ]
                print(f"Initialized Simulation with {len(self.image_files)} images from {source_path}")

    def get_frame_and_detections(self):
        frame = None
        if self.is_camera:
            import cv2
            if not self.cap or not self.cap.isOpened():
                return None, []
            success, frame = self.cap.read()
            if not success:
                return None, []
        else:
            if not self.image_files:
                return None, []
            import cv2
            # Pick a random image for each call to simulate "live" testing with random samples
            img_path = random.choice(self.image_files)
            frame = cv2.imread(str(img_path))
            if frame is None:
                return None, []

        # Run inference with NMS tightening for small PPE objects
        results = self.model(
            frame, 
            verbose=False,
            conf=0.1,       # Lowered to allow MIN_CONF filtering (especially for smoke/fire)
            iou=0.45,
            augment=False,  # Disabled for stability (prevents confidence flickering)
            imgsz=768       # Match training resolution
        )[0]
        
        h, w = frame.shape[:2]
        filtered_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            
            # Confidence filtering
            if conf < MIN_CONF.get(class_name, 0.45):
                continue
                
            filtered_detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [
                    float(x1 / w), 
                    float(y1 / h), 
                    float(x2 / w), 
                    float(y2 / h)
                ]
            })

        # Apply Hardened Logic Engine
        analyzed_detections = SafetyLogicEngine.associate_and_analyze(filtered_detections)

        return frame, analyzed_detections

    def release(self):
        if self.is_camera and self.cap and self.cap.isOpened():
            self.cap.release()
