"""YOLO Bounding Box Annotation Verifier module.

Scans YOLO annotation directories to verify whether valid bounding-box label files exist
before initiating model training. Enforces strict zero-fabrication policies.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ml.preprocessing.yolo_annotator import validate_yolo_label_line, YOLO_CLASSES

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class AnnotationVerifier:
    """Verifier engine scanning YOLO image and label directories."""

    def __init__(self, annotation_dir: Path):
        """Initialize AnnotationVerifier.

        Args:
            annotation_dir: Path to base annotations directory (e.g. data/annotations or data/annotations/yolo)
        """
        base_path = Path(annotation_dir)
        if base_path.name == "yolo":
            self.yolo_dir = base_path
        else:
            self.yolo_dir = base_path / "yolo"

    def verify_annotations(self) -> Dict[str, Any]:
        """Perform comprehensive inspection of YOLO label files across train, val, and test splits.

        Returns:
            Dict containing:
                - is_ready_for_training: bool
                - total_images: int
                - total_label_files: int
                - total_bounding_boxes: int
                - missing_annotations_count: int
                - missing_label_files: List[str]
                - invalid_label_lines: List[str]
                - split_breakdown: Dict[str, Dict[str, int]]
        """
        splits = ["train", "val", "test"]
        split_breakdown: Dict[str, Dict[str, int]] = {}
        missing_label_files: List[str] = []
        invalid_label_lines: List[str] = []

        total_images = 0
        total_label_files = 0
        total_bounding_boxes = 0

        for split in splits:
            img_dir = self.yolo_dir / "images" / split
            lbl_dir = self.yolo_dir / "labels" / split

            img_count = 0
            lbl_count = 0
            bbox_count = 0
            missing_in_split = 0

            if img_dir.exists() and img_dir.is_dir():
                for root, _, files in os.walk(img_dir):
                    for fname in files:
                        ext = Path(fname).suffix.lower()
                        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                            continue

                        img_count += 1
                        total_images += 1

                        stem = Path(fname).stem
                        expected_lbl_path = lbl_dir / f"{stem}.txt"

                        if not expected_lbl_path.exists():
                            missing_label_files.append(
                                str(expected_lbl_path.relative_to(self.yolo_dir.parent.parent))
                            )
                            missing_in_split += 1
                        else:
                            lbl_count += 1
                            total_label_files += 1

                            # Inspect label file contents
                            try:
                                with open(expected_lbl_path, "r", encoding="utf-8") as f:
                                    lines = f.readlines()

                                if not lines:
                                    # Empty label file = 0 boxes
                                    pass

                                for line_idx, line in enumerate(lines, start=1):
                                    line_str = line.strip()
                                    if not line_str:
                                        continue

                                    is_valid, msg = validate_yolo_label_line(line_str)
                                    if is_valid:
                                        bbox_count += 1
                                        total_bounding_boxes += 1
                                    else:
                                        invalid_label_lines.append(
                                            f"{expected_lbl_path.name}:{line_idx} - {msg}"
                                        )
                            except Exception as e:
                                invalid_label_lines.append(
                                    f"{expected_lbl_path.name} - Read error: {str(e)}"
                                )

            split_breakdown[split] = {
                "images_count": img_count,
                "label_files_count": lbl_count,
                "bounding_boxes_count": bbox_count,
                "missing_labels_count": missing_in_split,
            }

        # Training requires:
        # 1. At least 1 image file exists
        # 2. At least 1 valid label file with bounding boxes exists
        # 3. Zero invalid label syntax errors
        # 4. No missing label files for existing images
        is_ready = (
            total_images > 0
            and total_label_files > 0
            and total_bounding_boxes > 0
            and len(missing_label_files) == 0
            and len(invalid_label_lines) == 0
        )

        return {
            "is_ready_for_training": is_ready,
            "total_images": total_images,
            "total_label_files": total_label_files,
            "total_bounding_boxes": total_bounding_boxes,
            "missing_annotations_count": len(missing_label_files),
            "missing_label_files": missing_label_files,
            "invalid_label_lines_count": len(invalid_label_lines),
            "invalid_label_lines": invalid_label_lines,
            "split_breakdown": split_breakdown,
        }
