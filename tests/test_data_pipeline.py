"""Tests for the data pipeline and postprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.balance_dataset import load_yolo_labels
from src.inference.postprocess import PostProcessor, Detection
from src.config import SAFETY_CLASSES, CLASS_TO_IDX, ALERT_PRIORITY


def test_load_yolo_labels(tmp_path):
    lbl_file = tmp_path / "test.txt"
    lbl_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2\n")

    labels = load_yolo_labels(lbl_file)
    assert len(labels) == 2
    assert labels[0][0] == 0
    assert labels[1][0] == 1


def test_load_yolo_labels_missing_file():
    labels = load_yolo_labels(Path("/nonexistent/label.txt"))
    assert labels == []


def test_class_mappings():
    """Verify SAFETY_CLASSES and CLASS_TO_IDX are consistent."""
    assert len(SAFETY_CLASSES) == 22
    assert len(CLASS_TO_IDX) == 22
    for idx, name in enumerate(SAFETY_CLASSES):
        assert CLASS_TO_IDX[name] == idx


def test_hardhat_variants():
    """hardhat_on and hardhat_off are separate classes."""
    assert "hardhat_on" in CLASS_TO_IDX
    assert "hardhat_off" in CLASS_TO_IDX
    assert "vest_on" in CLASS_TO_IDX
    assert "vest_off" in CLASS_TO_IDX
    assert CLASS_TO_IDX["hardhat_on"] != CLASS_TO_IDX["hardhat_off"]
    assert CLASS_TO_IDX["vest_on"] != CLASS_TO_IDX["vest_off"]


def test_postprocess_confidence_filter():
    pp = PostProcessor(conf_thresh=0.5)

    detections = [
        {"class": "helmet", "class_id": 5, "confidence": 0.9, "bbox": [0, 0, 100, 100], "priority": "MEDIUM"},
        {"class": "hardhat_off", "class_id": 10, "confidence": 0.3, "bbox": [0, 0, 100, 100], "priority": "HIGH"},
    ]

    filtered = pp.filter_by_confidence(detections)
    assert len(filtered) == 1
    assert filtered[0]["class"] == "helmet"


def test_postprocess_nms():
    pp = PostProcessor(iou_thresh=0.4)

    detections = [
        {"class": "helmet", "class_id": 5, "confidence": 0.9, "bbox": [0, 0, 50, 50], "priority": "MEDIUM"},
        {"class": "helmet", "class_id": 5, "confidence": 0.8, "bbox": [5, 5, 55, 55], "priority": "MEDIUM"},
    ]

    kept = pp.apply_nms(detections)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9  # higher confidence kept


def test_postprocess_nms_different_classes():
    pp = PostProcessor(iou_thresh=0.4)

    detections = [
        {"class": "helmet", "class_id": 5, "confidence": 0.9, "bbox": [0, 0, 50, 50], "priority": "MEDIUM"},
        {"class": "vest_on", "class_id": 13, "confidence": 0.8, "bbox": [5, 5, 55, 55], "priority": "MEDIUM"},
    ]

    kept = pp.apply_nms(detections)
    assert len(kept) == 2  # different classes, both kept


def test_highest_priority():
    pp = PostProcessor()

    detections = [
        {"class": "helmet", "class_id": 5, "confidence": 0.9, "bbox": [0, 0, 100, 100], "priority": "MEDIUM"},
        {"class": "vest_on", "class_id": 13, "confidence": 0.8, "bbox": [0, 0, 100, 100], "priority": "MEDIUM"},
    ]
    assert pp.highest_priority(detections) == "MEDIUM"

    detections.append({"class": "hardhat_off", "class_id": 10, "confidence": 0.95, "bbox": [0, 0, 100, 100], "priority": "HIGH"})
    assert pp.highest_priority(detections) == "HIGH"

    assert pp.highest_priority([]) == "OK"


def test_to_detection_objs():
    pp = PostProcessor()
    raw = [
        {"class": "helmet", "class_id": 5, "confidence": 0.87, "bbox": [10, 20, 110, 120], "priority": "MEDIUM"},
    ]
    objs = pp.to_detection_objs(raw)
    assert len(objs) == 1
    assert isinstance(objs[0], Detection)
    assert objs[0].class_name == "helmet"
    assert objs[0].confidence == 0.87


def test_alert_priorities():
    """Verify alert priorities are set for all known classes."""
    for cls_name in SAFETY_CLASSES:
        assert cls_name in ALERT_PRIORITY, f"Missing alert priority for {cls_name}"
    # No CRITICAL class in this dataset (no fire/fall_risk/vehicle_proximity)
    # Violations are HIGH
    assert ALERT_PRIORITY["hardhat_off"] == "HIGH"
    assert ALERT_PRIORITY["vest_off"] == "HIGH"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
