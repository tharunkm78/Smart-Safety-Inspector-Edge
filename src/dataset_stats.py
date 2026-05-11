import os
from pathlib import Path
from collections import Counter
from config import COMBINED_DATA_DIR, SAFETY_CLASSES

def count_labels(label_dir):
    counts = Counter()
    if not label_dir.exists():
        return counts
    
    for lbl_file in label_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_idx = int(parts[0])
                    counts[cls_idx] += 1
    return counts

def main():
    print("=== Combined Dataset Statistics ===")
    if not COMBINED_DATA_DIR.exists():
        print(f"Error: Combined dataset not found at {COMBINED_DATA_DIR}")
        return

    splits = ["train", "valid", "test"]
    total_counts = Counter()

    for split in splits:
        lbl_dir = COMBINED_DATA_DIR / "labels" / split
        img_dir = COMBINED_DATA_DIR / "images" / split
        
        counts = count_labels(lbl_dir)
        total_counts.update(counts)
        
        img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        
        print(f"\nSplit: {split.upper()}")
        print(f"  Images:      {img_count}")
        print(f"  Annotations: {sum(counts.values())}")
        
        for idx, name in enumerate(SAFETY_CLASSES):
            count = counts.get(idx, 0)
            print(f"    [{idx}] {name:15s}: {count}")

    print("\n" + "="*35)
    print("OVERALL TOTALS")
    print(f"Total Annotations: {sum(total_counts.values())}")
    for idx, name in enumerate(SAFETY_CLASSES):
        count = total_counts.get(idx, 0)
        print(f"  [{idx}] {name:15s}: {count}")

if __name__ == "__main__":
    main()
