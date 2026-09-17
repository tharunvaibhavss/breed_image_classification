"""Unit and integration tests for complete end-to-end AI Breed Recognition Pipeline."""

import numpy as np
import pytest
from pathlib import Path

from ml.pipeline.inference_pipeline import BreedRecognitionPipeline, InferenceResult
from ml.detection.yolo_detector import DetectionResult


def test_pipeline_end_to_end_success():
    """Verify complete end-to-end inference pipeline execution and DTO schema."""
    pipeline = BreedRecognitionPipeline()
    sample_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    res = pipeline.predict(sample_img, generate_gradcam=True)

    assert isinstance(res, InferenceResult)
    assert res.animal_type in ["cattle", "buffalo", "unknown"]
    assert 0.0 <= res.animal_confidence <= 1.0
    assert len(res.bounding_box) == 4
    assert res.predicted_breed in ["Gir", "Ongole", "Sahiwal", "Jaffarabadi", "Murrah", "Surti"]
    assert 0.0 <= res.breed_confidence <= 1.0
    assert len(res.top_3_predictions) == 3
    assert res.prediction_status in [
        "success",
        "no_animal_detected",
        "multiple_animals_detected",
        "low_detection_confidence",
        "low_classification_confidence",
    ]
    assert "total_ms" in res.inference_time
    assert res.inference_time["total_ms"] >= 0.0
    assert res.model_versions["yolo_detector"] == "YOLOv8n-AnimalDetection-v1.0"


def test_pipeline_invalid_image_handling():
    """Verify handling of invalid or unreadable image input."""
    pipeline = BreedRecognitionPipeline()

    res = pipeline.predict("non_existent_file.jpg")

    assert res.prediction_status == "invalid_image"
    assert res.animal_type == "unknown"
    assert res.animal_confidence == 0.0
    assert res.breed_confidence == 0.0
    assert len(res.top_3_predictions) == 0


def test_pipeline_zero_detection_handling(tmp_path: Path):
    """Verify status code when YOLO detects zero animals."""
    pipeline = BreedRecognitionPipeline()
    # Pure black image
    blank_img = np.zeros((200, 200, 3), dtype=np.uint8)

    res = pipeline.predict(blank_img, generate_gradcam=False)

    assert res.prediction_status in ["no_animal_detected", "low_detection_confidence"]
    assert res.bounding_box == (0, 0, 200, 200)


def test_pipeline_multiple_animals_handling():
    """Verify handling when multiple animals are detected by YOLO."""
    pipeline = BreedRecognitionPipeline()
    sample_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

    # Mock detection of multiple animals
    det1 = DetectionResult(class_id=0, class_name="cattle", confidence=0.92, bbox=(10, 10, 150, 150))
    det2 = DetectionResult(class_id=1, class_name="buffalo", confidence=0.88, bbox=(200, 200, 350, 350))

    pipeline.yolo_detector.detect_animals = lambda source, conf_threshold=0.25: [det1, det2]

    res = pipeline.predict(sample_img, generate_gradcam=False)

    assert res.prediction_status == "multiple_animals_detected"
    assert res.animal_type == "cattle"
    assert res.animal_confidence == 0.92
    assert res.bounding_box == (10, 10, 150, 150)


def test_pipeline_low_classification_confidence():
    """Verify low classification confidence flag handling."""
    pipeline = BreedRecognitionPipeline()
    sample_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    det1 = DetectionResult(class_id=0, class_name="cattle", confidence=0.95, bbox=(10, 10, 200, 200))
    pipeline.yolo_detector.detect_animals = lambda source, conf_threshold=0.25: [det1]

    # Force low classification confidence < 0.20
    res = pipeline.predict(sample_img, classification_conf_threshold=0.99, generate_gradcam=False)

    assert res.prediction_status == "low_classification_confidence"
