"""Analyze dataset class distribution.

Usage:
    python -m src.data.dataset_stats
"""

import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, SAFETY_CLASSES


def count_yolo_labels(label_dir: Path) -> Counter:
    """Count class occurrences in a YOLO-format labels directory."""
    counts = Counter()
    if not label_dir.exists():
        return counts
    for i, lbl_file in enumerate(label_dir.rglob("*.txt")):
        if (i + 1) % 1000 == 0:
            print(f"    Counting labels: {i + 1} ...")
        with open(lbl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


def load_dataset_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def print_ds_stats(ds_dir: Path):
    """Print stats for a single dataset directory."""
    yaml_path = ds_dir / "data.yaml"
    if not yaml_path.exists():
        yaml_path = ds_dir / "dataset.yaml"
    
    meta = load_dataset_yaml(yaml_path)
    # Get class names from yaml or fallback to SAFETY_CLASSES if it's the combined one
    if "combined" in ds_dir.name or yaml_path.name == "dataset.yaml":
        raw_class_names = SAFETY_CLASSES
    else:
        raw_class_names = meta.get("names", []) if meta else []

    total_split_counts = Counter()
    # Check common YOLO split folders
    found_splits = False
    for split in ["train", "valid", "test"]:
        # Try both ds_dir/split/labels and ds_dir/labels/split
        lbl_dir = ds_dir / split / "labels"
        if not lbl_dir.exists():
            lbl_dir = ds_dir / "labels" / split
        
        if not lbl_dir.exists():
            continue
            
        found_splits = True
        counts = count_yolo_labels(lbl_dir)
        total_split_counts.update(counts)
        split_total = sum(counts.values())
        print(f"  [{split}] annotations: {split_total}")
        for cls_idx, count in sorted(counts.items()):
            name = raw_class_names[cls_idx] if cls_idx < len(raw_class_names) else f"class_{cls_idx}"
            print(f"    [{cls_idx:2d}] {name:25s}: {count}")

    if not found_splits:
        # Maybe it's just a bunch of files in the root?
        counts = count_yolo_labels(ds_dir)
        if counts:
            print(f"  [root] annotations: {sum(counts.values())}")
            for cls_idx, count in sorted(counts.items()):
                name = raw_class_names[cls_idx] if cls_idx < len(raw_class_names) else f"class_{cls_idx}"
                print(f"    [{cls_idx:2d}] {name:25s}: {count}")
            total_split_counts = counts

    print(f"  [total] annotations: {sum(total_split_counts.values())}")
    print()


def print_stats():
    print("=== Dataset Statistics ===\n")

    raw_dir = DATA_DIR / "raw"
    combined_dir = DATA_DIR / "combined"

    if raw_dir.exists():
        print("--- Raw Datasets ---")
        for ds_dir in sorted(raw_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            print(f"Dataset: {ds_dir.name}")
            print_ds_stats(ds_dir)

    if combined_dir.exists():
        print("\n--- Combined & Balanced Dataset ---")
        print_ds_stats(combined_dir)

    # Show SAFETY_CLASSES for the combined model
    print(f"Combined model classes ({len(SAFETY_CLASSES)}):")
    for i, c in enumerate(SAFETY_CLASSES):
        print(f"  {i}: {c}")


if __name__ == "__main__":
    try:
        print_stats()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
