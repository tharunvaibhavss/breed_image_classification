"""Unit tests for YOLO label validation and directory hierarchy setup."""

from pathlib import Path
from ml.preprocessing.yolo_annotator import (
    validate_yolo_label_line,
    YOLOAnnotatorSetup,
    YOLO_CLASSES,
)


def test_valid_yolo_label_lines():
    """Test validation of valid YOLO bounding box annotation lines."""
    # class 0 = cattle, normalized coords
    valid_line_0 = "0 0.5 0.5 0.4 0.6"
    is_valid, msg = validate_yolo_label_line(valid_line_0)
    assert is_valid is True

    # class 1 = buffalo, boundary coordinates
    valid_line_1 = "1 0.0 1.0 1.0 0.0"
    is_valid, msg = validate_yolo_label_line(valid_line_1)
    assert is_valid is True


def test_invalid_yolo_label_lines():
    """Test catching invalid class_ids, coordinate bounds, and malformed lines."""
    # Invalid class_id 2
    is_valid, msg = validate_yolo_label_line("2 0.5 0.5 0.4 0.6")
    assert is_valid is False
    assert "Invalid class_id" in msg

    # Coordinate > 1.0
    is_valid, msg = validate_yolo_label_line("0 1.5 0.5 0.4 0.6")
    assert is_valid is False
    assert "out of bounds" in msg

    # Missing tokens
    is_valid, msg = validate_yolo_label_line("0 0.5 0.5")
    assert is_valid is False
    assert "Expected 5 tokens" in msg


def test_yolo_setup_directories_and_yaml(tmp_path: Path):
    """Test creation of YOLO dataset directories and yolo_config.yaml."""
    setup = YOLOAnnotatorSetup(base_annotation_dir=tmp_path)
    dirs = setup.setup_directories()

    assert dirs["images_train"].exists()
    assert dirs["labels_val"].exists()

    yaml_path = setup.create_dataset_yaml(project_root=tmp_path)
    assert yaml_path.exists()
    assert "yolo_config.yaml" in str(yaml_path)
