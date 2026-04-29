"""Tests for the inference engine (detector + postprocessing)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.postprocess import PostProcessor


def test_iou_calculation():
    pp = PostProcessor()

    # Two identical boxes
    box_a = [0, 0, 100, 100]
    box_b = [0, 0, 100, 100]
    assert pp._iou(box_a, box_b) == 1.0

    # Two non-overlapping boxes
    box_a = [0, 0, 50, 50]
    box_b = [60, 60, 100, 100]
    assert pp._iou(box_a, box_b) == 0.0

    # Two partially overlapping boxes
    box_a = [0, 0, 50, 50]
    box_b = [25, 25, 75, 75]
    iou = pp._iou(box_a, box_b)
    assert 0.1 < iou < 0.5


def test_detection_to_dict():
    from src.inference.postprocess import Detection

    det = Detection(
        class_name="helmet",
        class_id=5,
        confidence=0.8765,
        bbox=(10.0, 20.0, 110.0, 120.0),
        priority="MEDIUM",
    )

    d = det.to_dict()
    assert d["class"] == "helmet"
    assert d["confidence"] == 0.8765
    assert d["bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert d["priority"] == "MEDIUM"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
