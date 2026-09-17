"""YOLO Annotation workflow and label validation module.

Manages YOLO directory structures, dataset YAML configs, and bounding box format validators.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Standard YOLO Animal Detection Class Mapping
YOLO_CLASSES: Dict[int, str] = {
    0: "cattle",
    1: "buffalo",
}


def validate_yolo_label_line(line: str) -> Tuple[bool, str]:
    """Validate a single YOLO bounding box label line.

    Expected format: <class_id> <x_center> <y_center> <width> <height>
    Coordinates must be normalized floats in range [0.0, 1.0].

    Args:
        line: String content of a label line.

    Returns:
        Tuple of (is_valid, error_message).
    """
    parts = line.strip().split()
    if not parts:
        return True, ""  # Empty line / empty annotation is valid

    if len(parts) != 5:
        return False, f"Expected 5 tokens (<class_id> <x> <y> <w> <h>), got {len(parts)}"

    try:
        class_id = int(parts[0])
    except ValueError:
        return False, f"class_id must be integer, got '{parts[0]}'"

    if class_id not in YOLO_CLASSES:
        return False, f"Invalid class_id {class_id}. Expected one of {list(YOLO_CLASSES.keys())}"

    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return False, "Bounding box coordinates must be floating point numbers"

    x_center, y_center, width, height = coords

    for val, name in zip(
        coords, ["x_center", "y_center", "width", "height"]
    ):
        if not (0.0 <= val <= 1.0):
            return (
                False,
                f"Normalized coordinate '{name}' value {val} out of bounds [0.0, 1.0]",
            )

    return True, "Valid YOLO annotation"


class YOLOAnnotatorSetup:
    """Setup manager for YOLO detection annotations and directory hierarchy."""

    def __init__(self, base_annotation_dir: Path):
        self.base_dir = Path(base_annotation_dir)
        self.yolo_dir = self.base_dir / "yolo"

    def setup_directories(self) -> Dict[str, Path]:
        """Create YOLO directory hierarchy for train, val, and test splits."""
        dirs = {
            "yolo_root": self.yolo_dir,
            "images_train": self.yolo_dir / "images" / "train",
            "images_val": self.yolo_dir / "images" / "val",
            "images_test": self.yolo_dir / "images" / "test",
            "labels_train": self.yolo_dir / "labels" / "train",
            "labels_val": self.yolo_dir / "labels" / "val",
            "labels_test": self.yolo_dir / "labels" / "test",
        }

        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    def create_dataset_yaml(self, project_root: Path) -> Path:
        """Generate yolo_config.yaml file describing detection dataset parameters."""
        self.setup_directories()
        yaml_path = self.yolo_dir / "yolo_config.yaml"

        config_data = {
            "path": str(self.yolo_dir.relative_to(project_root)),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": YOLO_CLASSES,
            "nc": len(YOLO_CLASSES),
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, sort_keys=False, indent=2)

        return yaml_path
