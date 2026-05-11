import os
import random
import shutil
import yaml
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    RAW_DATA_DIR, COMBINED_DATA_DIR, SAFETY_CLASSES, 
    CLASS_TO_IDX, DATASET_MAPPINGS, NUM_CLASSES
)

class DatasetPreparer:
    def __init__(self):
        self.samples = []
        
    def load_yolo_labels(self, label_path):
        if not label_path.exists(): return []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    labels.append([float(x) for x in parts])
        return labels

    def save_yolo_labels(self, label_path, labels):
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            for lbl in labels:
                f.write(f"{int(lbl[0])} {' '.join(f'{x:.6f}' for x in lbl[1:])}\n")

    def collect_samples(self):
        for dataset_name, mapping in DATASET_MAPPINGS.items():
            ds_dir = RAW_DATA_DIR / dataset_name
            if not ds_dir.exists():
                print(f"Skipping {dataset_name}, not found.")
                continue

            print(f"Processing raw dataset: {dataset_name}")
            for split in ["train", "valid", "test", ""]:
                img_dir = ds_dir / split / "images" if split else ds_dir / "images"
                lbl_dir = ds_dir / split / "labels" if split else ds_dir / "labels"
                
                if not img_dir.exists(): continue
                
                for img_path in img_dir.glob("*.jpg"):
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    raw_labels = self.load_yolo_labels(lbl_path)
                    
                    mapped_labels = []
                    image_classes = set()
                    for lbl in raw_labels:
                        raw_cls = int(lbl[0])
                        if raw_cls in mapping:
                            target_name = mapping[raw_cls]
                            if target_name in CLASS_TO_IDX:
                                target_cls = CLASS_TO_IDX[target_name]
                                mapped_labels.append([target_cls] + lbl[1:])
                                image_classes.add(target_cls)
                            
                    self.samples.append({
                        "image": img_path,
                        "labels": mapped_labels,
                        "classes": image_classes
                    })

    def balance_samples(self):
        print("\nBalancing dataset (Multi-label Aware)...")
        
        # 1. First, sort samples by "rarity" or priority
        # Fire/Smoke should be capped to avoid dominance
        # Person should be maximized
        
        random.shuffle(self.samples)
        
        balanced_samples = []
        counts = Counter()
        
        # Target ranges from user feedback
        targets = {
            "person": 7000,
            "helmet_on": 5000,
            "vest_on": 5000,
            "gloves_on": 5000,
            "boots": 5000,
            "fire": 3500,
            "smoke": 3500
        }
        
        # Simple greedy approach for multi-label balancing:
        # If an image contains a class that is under its target, keep it.
        for s in self.samples:
            if not s["labels"]:
                # Keep some background images
                if counts["background"] < 500:
                    balanced_samples.append(s)
                    counts["background"] += 1
                continue
                
            should_keep = False
            for cls_idx in s["classes"]:
                cls_name = SAFETY_CLASSES[cls_idx]
                if counts[cls_name] < targets.get(cls_name, 5000):
                    should_keep = True
                    break
            
            if should_keep:
                balanced_samples.append(s)
                for cls_idx in s["classes"]:
                    counts[SAFETY_CLASSES[cls_idx]] += 1

        print("\nFinal Balanced Participation:")
        for name in SAFETY_CLASSES:
            print(f"  {name:15s}: {counts[name]}")
        print(f"  backgrounds    : {counts['background']}")
        
        return balanced_samples

    def build_combined_dataset(self, samples):
        if COMBINED_DATA_DIR.exists():
            shutil.rmtree(COMBINED_DATA_DIR)
            
        random.shuffle(samples)
        n = len(samples)
        splits = {
            "train": samples[:int(n * 0.8)],
            "valid": samples[int(n * 0.8):int(n * 0.95)],
            "test": samples[int(n * 0.95):]
        }
        
        print("\nWriting combined dataset...")
        for split_name, split_samples in splits.items():
            print(f"  Writing {split_name} split: {len(split_samples)} images")
            img_out = COMBINED_DATA_DIR / "images" / split_name
            lbl_out = COMBINED_DATA_DIR / "labels" / split_name
            img_out.mkdir(parents=True)
            lbl_out.mkdir(parents=True)
            
            for i, s in enumerate(split_samples):
                dst_img = img_out / f"img_{split_name}_{i:06d}{s['image'].suffix}"
                dst_lbl = lbl_out / f"img_{split_name}_{i:06d}.txt"
                shutil.copy2(s["image"], dst_img)
                self.save_yolo_labels(dst_lbl, s["labels"])

        # Write YAML
        yaml_path = COMBINED_DATA_DIR / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({
                "path": str(COMBINED_DATA_DIR),
                "train": "images/train",
                "val": "images/valid",
                "test": "images/test",
                "names": {idx: name for idx, name in enumerate(SAFETY_CLASSES)}
            }, f, sort_keys=False)
            
        print(f"\nDone. Combined dataset created at {COMBINED_DATA_DIR}")

if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.collect_samples()
    balanced = preparer.balance_samples()
    preparer.build_combined_dataset(balanced)
