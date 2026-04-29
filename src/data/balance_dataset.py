"""Balance and merge the PPE and Construction Equipment datasets.

Reads class names directly from each dataset's data.yaml, maps raw class IDs
to SAFETY_CLASSES indices, oversamples minority classes, and writes the
combined dataset to data/combined/ in YOLO format.

Usage:
    python -m src.data.balance_dataset
"""

import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_DIR, PPE_DIR, CONSTRUCTION_DIR,
    SAFETY_CLASSES, CLASS_TO_IDX,
    MIN_SAMPLES_PER_CLASS, MAX_SAMPLES_PER_CLASS, NUM_CLASSES,
)


# ─── Raw class ID maps ─────────────────────────────────────────────────────────
# PPE dataset (ppe-v2): IDs 0-7 → SAFETY_CLASSES 0-7
PPE_ID_MAP = {
    0: CLASS_TO_IDX["dust_mask"],   # Dust mask
    1: CLASS_TO_IDX["eye_wear"],     # Eye Wear
    2: CLASS_TO_IDX["glove"],        # Glove
    3: CLASS_TO_IDX["jacket"],       # Jacket
    4: CLASS_TO_IDX["boots"],         # Protective Boots
    5: CLASS_TO_IDX["helmet"],       # Protective Helmet
    6: CLASS_TO_IDX["shield"],        # Shield
    7: CLASS_TO_IDX["person"],       # people
}

# Construction dataset: IDs 0-13 → SAFETY_CLASSES 8-21
CONSTRUCTION_ID_MAP = {
    0: CLASS_TO_IDX["dump_truck"],      # Dump Truck
    1: CLASS_TO_IDX["excavator"],       # Excavator
    2: CLASS_TO_IDX["front_end_loader"],# Front End Loader
    3: CLASS_TO_IDX["gloves"],          # Gloves (construction)
    4: CLASS_TO_IDX["hardhat_off"],     # Hard Hat OFF
    5: CLASS_TO_IDX["hardhat_on"],      # Hard Hat ON
    6: CLASS_TO_IDX["ladder"],          # Ladder
    7: CLASS_TO_IDX["vest_off"],        # Safety Vest OFF
    8: CLASS_TO_IDX["vest_on"],         # Safety Vest ON
    9: CLASS_TO_IDX["skid_steer"],      # Skid Steer
    10: CLASS_TO_IDX["tractor_trailer"], # Tractor Trailer
    11: CLASS_TO_IDX["trailer"],        # Trailer
    12: CLASS_TO_IDX["vehicle"],        # Vehicle
    13: CLASS_TO_IDX["worker"],         # Worker
}


def load_yolo_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    with open(label_path, "r") as f:
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                labels.append([float(x) for x in parts])
        return labels


def save_yolo_labels(label_path: Path, labels: list[list[float]]):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for lbl in labels:
            # Ensure class_id is an integer
            cls_id = int(lbl[0])
            f.write(f"{cls_id} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")


def collect_samples_from_split(img_dir: Path, lbl_dir: Path, id_map: dict) -> list[dict]:
    """Collect all image/label pairs from a train/valid/test split.

    Args:
        img_dir: path/to/{split}/images
        lbl_dir: path/to/{split}/labels
        id_map: {raw_class_id: safety_class_idx}
    """
    samples = []
    if not img_dir.exists():
        return samples

    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        raw_labels = load_yolo_labels(lbl_path)

        # Remap class IDs to SAFETY_CLASSES indices
        remapped = []
        unmapped = 0
        for lbl in raw_labels:
            raw_cls = int(lbl[0])
            safety_cls = id_map.get(raw_cls)
            if safety_cls is not None:
                remapped.append([float(safety_cls)] + lbl[1:])
            else:
                unmapped += 1

        if unmapped:
            print(f"    WARNING: {unmapped} annotations with unmapped class IDs in {img_path.name}")

        samples.append({
            "image": img_path,
            "labels": remapped,
            "num_annots": len(remapped),
        })
    return samples


def collect_all_samples() -> list[dict]:
    """Collect all samples from both datasets across all splits."""
    all_samples = []

    for split in ["train", "valid", "test"]:
        # PPE
        img_dir = PPE_DIR / split / "images"
        lbl_dir = PPE_DIR / split / "labels"
        n = len(all_samples)
        all_samples.extend(collect_samples_from_split(img_dir, lbl_dir, PPE_ID_MAP))
        if len(all_samples) > n:
            print(f"  PPE/{split}: {len(all_samples) - n} samples")

        # Construction
        img_dir = CONSTRUCTION_DIR / split / "images"
        lbl_dir = CONSTRUCTION_DIR / split / "labels"
        n = len(all_samples)
        all_samples.extend(collect_samples_from_split(img_dir, lbl_dir, CONSTRUCTION_ID_MAP))
        if len(all_samples) > n:
            print(f"  Construction/{split}: {len(all_samples) - n} samples")

    return all_samples


def compute_class_counts(samples: list[dict]) -> Counter:
    """Count annotations per SAFETY_CLASSES index."""
    counts = Counter()
    for s in samples:
        for lbl in s["labels"]:
            counts[int(lbl[0])] += 1
    return counts


def oversample(samples: list[dict], target_counts: dict) -> list[dict]:
    """Oversample samples so each class reaches target_counts."""
    # Group samples by dominant class
    by_class = defaultdict(list)
    for s in samples:
        if s["num_annots"] == 0:
            by_class["_empty"].append(s)
            continue
        first_cls = int(s["labels"][0][0])
        by_class[first_cls].append(s)

    balanced = []
    for cls_idx in range(NUM_CLASSES):
        cls_samples = by_class.get(cls_idx, [])
        target = target_counts.get(cls_idx, 0)

        if not cls_samples:
            if target > 0:
                print(f"  WARNING: class {cls_idx} ({SAFETY_CLASSES[cls_idx]}) has 0 samples but needs {target}")
            continue

        n = len(cls_samples)
        if target <= n:
            balanced.extend(cls_samples)
        else:
            copies = (target + n - 1) // n
            oversampled = cls_samples * copies
            balanced.extend(oversampled[:target])

    return balanced


def balance():
    print("=== Balancing Dataset ===\n")

    combined_dir = DATA_DIR / "combined"
    out_img_dir = combined_dir / "images" / "train"
    out_lbl_dir = combined_dir / "labels" / "train"

    if out_img_dir.exists():
        print(f"Balanced dataset already exists at {combined_dir}/.")
        print(f"Remove it to rebalance: Remove-Item -Recurse {combined_dir}")
        return

    print("Collecting samples from raw datasets...")
    all_samples = collect_all_samples()
    print(f"\nTotal samples: {len(all_samples)}")

    # Count per class
    raw_counts = compute_class_counts(all_samples)
    total_annots = sum(raw_counts.values())
    print(f"\nRaw class distribution ({total_annots} total annotations):")
    for cls_idx, count in sorted(raw_counts.items()):
        name = SAFETY_CLASSES[cls_idx]
        print(f"  [{cls_idx:2d}] {name:20s}: {count}")

    # Target counts: absolute cap + floor
    target_counts = {}
    for cls_idx in range(NUM_CLASSES):
        raw = raw_counts.get(cls_idx, 0)
        if raw == 0:
            target_counts[cls_idx] = MIN_SAMPLES_PER_CLASS
        else:
            # Cap at MAX_SAMPLES_PER_CLASS to avoid excessive oversampling
            target_counts[cls_idx] = min(raw, MAX_SAMPLES_PER_CLASS)

    total_target = sum(target_counts.values())
    print(f"\nTarget distribution (absolute cap {MAX_SAMPLES_PER_CLASS}, floor {MIN_SAMPLES_PER_CLASS}):")
    print(f"  Total samples after balancing: ~{total_target}")
    for cls_idx, target in sorted(target_counts.items()):
        raw = raw_counts.get(cls_idx, 0)
        tag = ""
        if raw == 0:   tag = " [added]"
        elif raw > MAX_SAMPLES_PER_CLASS: tag = " [capped]"
        print(f"  [{cls_idx:2d}] {SAFETY_CLASSES[cls_idx]:20s}: raw={raw:5d}  target={target}{tag}")

    # Oversample
    balanced = oversample(all_samples, target_counts)
    print(f"\nBalanced dataset: {len(balanced)} samples")

    # Write combined dataset
    print(f"\nWriting to {combined_dir}/ ...")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(balanced):
        new_name = f"img_{i:06d}{s['image'].suffix}"
        dst_img = out_img_dir / new_name
        dst_lbl = out_lbl_dir / f"img_{i:06d}.txt"

        shutil.copy2(s["image"], dst_img)
        save_yolo_labels(dst_lbl, s["labels"])

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(balanced)} ...")

    # Create dataset.yaml for YOLO training
    yaml_content = {
        "path": str(combined_dir),
        "train": "images/train",
        "val": "images/train",
        "names": {idx: name for idx, name in enumerate(SAFETY_CLASSES)},
    }
    import yaml
    yaml_path = combined_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    final_img_count = len(list(out_img_dir.glob("*.jpg")))
    print(f"\nDone.")
    print(f"  Combined dataset: {combined_dir}")
    print(f"  Images: {out_img_dir} ({final_img_count} images)")
    print(f"  Config: {yaml_path}")
    print(f"\nNext: python -m src.training.train_yolov8")


if __name__ == "__main__":
    balance()
