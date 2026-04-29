"""Convert annotations between COCO and YOLO formats.

Usage:
    python -m src.data.convert_annotations --input data/raw/ppe-v2 --output data/annotations/yolo/
    python -m src.data.convert_annotations --input data/annotations/coco.json --output data/annotations/yolo/
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import SAFETY_CLASSES, CLASS_TO_IDX


def coco_to_yolo(coco_json_path: Path, output_dir: Path, images_dir: Path):
    """Convert COCO JSON annotations to YOLO txt format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations_by_image = [[] for _ in range(max(images.keys()) + 1)]
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id < len(annotations_by_image):
            annotations_by_image[img_id].append(ann)

    for img_id, img_meta in images.items():
        img_w, img_h = img_meta["width"], img_meta["height"]
        img_stem = Path(img_meta["file_name"]).stem

        yolo_lines = []
        for ann in annotations_by_image[img_id]:
            cat_name = categories[ann["category_id"]]
            if cat_name not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[cat_name]

            bbox = ann["bbox"]  # [x, y, w, h] in absolute pixels
            x_center = (bbox[0] + bbox[2] / 2) / img_w
            y_center = (bbox[1] + bbox[3] / 2) / img_h
            w_norm = bbox[2] / img_w
            h_norm = bbox[3] / img_h

            yolo_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        lbl_path = output_dir / f"{img_stem}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))


def yolo_to_coco(yolo_dir: Path, images_dir: Path, class_names: list) -> dict:
    """Convert YOLO txt annotations to COCO JSON format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
    }

    ann_id = 1
    img_id = 1

    for img_path in sorted(images_dir.glob("*.jpg")):
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        lbl_path = yolo_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_idx, x_center, y_center, w_norm, h_norm = [float(x) for x in parts]
                    bbox = [
                        (x_center - w_norm / 2) * w,
                        (y_center - h_norm / 2) * h,
                        w_norm * w,
                        h_norm * h,
                    ]
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_idx),
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    })
                    ann_id += 1

        img_id += 1

    return coco


def main():
    parser = argparse.ArgumentParser(description="Convert between COCO and YOLO annotation formats")
    parser.add_argument("--input", type=Path, required=True, help="Input file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory")
    parser.add_argument("--images-dir", type=Path, default=None, help="Images directory (for YOLO input)")
    args = parser.parse_args()

    if args.input.suffix == ".json":
        print(f"Converting COCO → YOLO: {args.input} → {args.output}")
        coco_to_yolo(args.input, args.output, args.images_dir or args.input.parent)
        print("Done.")
    else:
        print(f"Converting YOLO → COCO: {args.input} → {args.output}")
        coco = yolo_to_coco(args.input, args.images_dir or args.input, SAFETY_CLASSES)
        with open(args.output, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"Done. {len(coco['images'])} images, {len(coco['annotations'])} annotations.")


if __name__ == "__main__":
    main()
