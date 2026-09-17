"""Unit tests for YOLO animal detection, annotation verifier, and ROI cropping logic."""

import numpy as np
import pytest
from pathlib import Path

from ml.detection.annotation_verifier import AnnotationVerifier
from ml.detection.yolo_detector import YOLOAnimalDetector, DetectionResult
from ml.preprocessing.yolo_annotator import YOLOAnnotatorSetup


def test_annotation_verifier_incomplete_labels_detection(tmp_path: Path):
    """Verify AnnotationVerifier catches missing label files and stops training."""
    setup = YOLOAnnotatorSetup(base_annotation_dir=tmp_path)
    dirs = setup.setup_directories()

    # Create dummy image without label file in train split
    dummy_img = dirs["images_train"] / "cattle_01.jpg"
    dummy_img.write_bytes(b"DUMMY_IMAGE_BYTES")

    verifier = AnnotationVerifier(annotation_dir=tmp_path)
    report = verifier.verify_annotations()

    assert report["is_ready_for_training"] is False
    assert report["missing_annotations_count"] == 1
    assert len(report["missing_label_files"]) == 1


def test_annotation_verifier_valid_labels(tmp_path: Path):
    """Verify AnnotationVerifier reports readiness when valid label text files exist."""
    setup = YOLOAnnotatorSetup(base_annotation_dir=tmp_path)
    dirs = setup.setup_directories()

    # Create image file and matching valid label file
    dummy_img = dirs["images_train"] / "cattle_01.jpg"
    dummy_img.write_bytes(b"DUMMY_IMAGE_BYTES")

    dummy_lbl = dirs["labels_train"] / "cattle_01.txt"
    dummy_lbl.write_text("0 0.5 0.5 0.4 0.4\n")

    verifier = AnnotationVerifier(annotation_dir=tmp_path)
    report = verifier.verify_annotations()

    assert report["is_ready_for_training"] is True
    assert report["missing_annotations_count"] == 0
    assert report["total_bounding_boxes"] == 1


def test_detection_result_model():
    """Verify DetectionResult Pydantic model serialization."""
    det = DetectionResult(
        class_id=0,
        class_name="cattle",
        confidence=0.92,
        bbox=(10, 20, 150, 200),
    )

    d = det.to_dict()
    assert d["class_id"] == 0
    assert d["class_name"] == "cattle"
    assert d["confidence"] == 0.92
    assert d["bbox"] == (10, 20, 150, 200)


def test_crop_animals_single_and_multiple():
    """Verify animal ROI crop extraction and bounding box padding."""
    detector = YOLOAnimalDetector()
    img_rgb = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    det1 = DetectionResult(class_id=0, class_name="cattle", confidence=0.95, bbox=(50, 50, 200, 200))
    det2 = DetectionResult(class_id=1, class_name="buffalo", confidence=0.88, bbox=(250, 250, 450, 450))

    # Single detection crop test
    crops_1 = detector.crop_animals(img_rgb, [det1], pad_percent=0.0)
    assert len(crops_1) == 1
    assert crops_1[0]["crop"].shape == (150, 150, 3)
    assert crops_1[0]["class_name"] == "cattle"

    # Multiple detection crops test
    crops_2 = detector.crop_animals(img_rgb, [det1, det2], pad_percent=0.05)
    assert len(crops_2) == 2
    assert crops_2[0]["class_name"] == "cattle"
    assert crops_2[1]["class_name"] == "buffalo"


def test_crop_animals_zero_detections():
    """Verify handling when zero detections exist."""
    detector = YOLOAnimalDetector()
    img_rgb = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    crops = detector.crop_animals(img_rgb, [], pad_percent=0.0)
    assert len(crops) == 0
