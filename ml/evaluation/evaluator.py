"""Model Evaluator module for EfficientNet-B0 breed classification test set evaluation.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.classification.efficientnet import BreedClassifier
from ml.classification.metrics import calculate_classification_metrics, calculate_topk_accuracy

# 6 Breed Class Names ordered by class_id (0..5)
CLASS_NAMES: List[str] = [
    "Gir",
    "Ongole",
    "Sahiwal",
    "Jaffarabadi",
    "Murrah",
    "Surti",
]


class ModelEvaluator:
    """Evaluator engine executing test set inference and metric calculations."""

    def __init__(self, num_classes: int = 6, device: Optional[str] = None):
        """Initialize ModelEvaluator.

        Args:
            num_classes: Number of classes (default 6).
            device: Compute device ('cpu' or 'cuda:0').
        """
        self.num_classes = num_classes
        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute 6x6 confusion matrix.

        Args:
            y_true: True integer class labels (N,).
            y_pred: Predicted integer class labels (N,).

        Returns:
            6x6 integer numpy array confusion matrix.
        """
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                cm[t, p] += 1
        return cm

    def compute_per_class_metrics(
        self, cm: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Compute Precision, Recall, F1-Score, and Support for each breed class.

        Args:
            cm: 6x6 confusion matrix array.

        Returns:
            Dict mapping breed name to per-class metrics dictionary.
        """
        per_class: Dict[str, Dict[str, Any]] = {}

        for c in range(self.num_classes):
            breed_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            support = int(np.sum(cm[c, :]))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[breed_name] = {
                "class_id": c,
                "breed_name": breed_name,
                "precision": round(float(precision) * 100.0, 2),
                "recall": round(float(recall) * 100.0, 2),
                "f1_score": round(float(f1) * 100.0, 2),
                "support": support,
            }

        return per_class

    def evaluate(
        self,
        test_loader: DataLoader,
        model: Optional[nn.Module] = None,
        model_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute test set evaluation and return complete metrics dictionary.

        Args:
            test_loader: PyTorch DataLoader for test split.
            model: Optional instantiated BreedClassifier model.
            model_path: Optional path to saved model weights checkpoint.

        Returns:
            Dict containing overall metrics, per-class metrics, confusion matrix, and predictions.
        """
        if model is None:
            model = BreedClassifier(num_classes=self.num_classes, pretrained=False)
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []

        if len(test_loader.dataset) == 0:
            # Empty test set fallback
            dummy_cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            return {
                "status": "warning",
                "message": "Test set empty. Baseline evaluation metrics returned.",
                "total_test_samples": 0,
                "test_loss": 0.0,
                "accuracy": 0.0,
                "top_3_accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "confusion_matrix": dummy_cm.tolist(),
                "per_class_metrics": self.compute_per_class_metrics(dummy_cm),
                "y_true": [],
                "y_pred_probs": [],
            }

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = model(images)
                loss = criterion(logits, targets)

                test_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_targets.extend(targets.cpu().numpy().tolist())

        total_samples = len(test_loader.dataset)
        test_loss /= total_samples
        y_probs_np = np.vstack(all_probs)
        y_true_np = np.array(all_targets, dtype=int)
        y_pred_np = np.argmax(y_probs_np, axis=1)

        # Compute overall metrics
        overall_metrics = calculate_classification_metrics(
            y_true=y_true_np, y_pred_probs=y_probs_np, num_classes=self.num_classes
        )

        # Compute confusion matrix and per-class metrics
        cm = self.compute_confusion_matrix(y_true_np, y_pred_np)
        per_class = self.compute_per_class_metrics(cm)

        return {
            "status": "success",
            "total_test_samples": total_samples,
            "test_loss": round(float(test_loss), 4),
            "accuracy": overall_metrics["accuracy"],
            "top_3_accuracy": overall_metrics["top_3_accuracy"],
            "precision_macro": overall_metrics["precision_macro"],
            "recall_macro": overall_metrics["recall_macro"],
            "f1_macro": overall_metrics["f1_macro"],
            "f1_weighted": overall_metrics["f1_weighted"],
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class,
            "y_true": y_true_np.tolist(),
            "y_pred_probs": y_probs_np.tolist(),
        }
