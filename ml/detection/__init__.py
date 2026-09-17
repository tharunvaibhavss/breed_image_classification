"""ML Detection package for YOLO animal detection, ROI cropping, and annotation verification."""

from ml.detection.annotation_verifier import AnnotationVerifier
from ml.detection.yolo_detector import YOLOAnimalDetector, DetectionResult
from ml.detection.yolo_trainer import YOLOModelTrainer

__all__ = [
    "AnnotationVerifier",
    "YOLOAnimalDetector",
    "DetectionResult",
    "YOLOModelTrainer",
]
