"""ML Classification package for EfficientNet-B0 breed identification."""

from ml.classification.efficientnet import BreedClassifier
from ml.classification.dataset import BreedDataset, create_dataloaders
from ml.classification.metrics import calculate_classification_metrics, calculate_topk_accuracy
from ml.classification.trainer import EfficientNetTrainer
from ml.classification.predictor import BreedPredictor, PredictionResult

__all__ = [
    "BreedClassifier",
    "BreedDataset",
    "create_dataloaders",
    "calculate_classification_metrics",
    "calculate_topk_accuracy",
    "EfficientNetTrainer",
    "BreedPredictor",
    "PredictionResult",
]
