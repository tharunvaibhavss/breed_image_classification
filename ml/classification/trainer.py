"""EfficientNet-B0 Model Trainer module with MLflow experiment tracking integration.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.config import settings
from ml.common.env_check import get_pytorch_environment_info
from ml.common.mlflow_tracker import MLflowTracker, EFFICIENTNET_EXPERIMENT_NAME
from ml.classification.efficientnet import BreedClassifier
from ml.classification.metrics import calculate_classification_metrics


def set_reproducible_seeds(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, PyTorch CPU, and PyTorch CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping handler to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = float("inf")
        self.early_stop = False

    def check(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class EfficientNetTrainer:
    """Trainer manager orchestrating EfficientNet-B0 breed classification transfer learning."""

    def __init__(
        self,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        models_dir: Optional[Path] = None,
        dataset_version: str = "dataset_v001",
    ):
        """Initialize EfficientNetTrainer."""
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.dataset_version = dataset_version
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = MLflowTracker()

        set_reproducible_seeds(seed)

        # Environment detection for CUDA / CPU selection
        env_info = get_pytorch_environment_info()
        self.device = torch.device("cuda:0" if env_info.get("cuda_available") else "cpu")

    def train(
        self,
        dataloaders: Dict[str, DataLoader],
        epochs: int = 20,
        batch_size: int = 16,
        patience: int = 7,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute model training loop.

        Args:
            dataloaders: Dict containing 'train', 'val', and 'test' DataLoaders.
            epochs: Max training epochs.
            batch_size: Batch size.
            patience: Early stopping patience epochs.
            class_weights: Optional tensor of class weights for loss function.

        Returns:
            Dict containing best metrics and model checkpoint paths.
        """
        model = BreedClassifier(num_classes=self.num_classes, pretrained=True)
        model = model.to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stopping = EarlyStopping(patience=patience)

        best_val_f1 = 0.0
        best_model_path = self.models_dir / "efficientnet_best.pth"
        last_model_path = self.models_dir / "efficientnet_last.pth"

        train_loader = dataloaders.get("train")
        val_loader = dataloaders.get("val")

        if not train_loader or len(train_loader.dataset) == 0:
            # If dataset is empty, save initialized baseline checkpoint safely
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), last_model_path)
            return {
                "status": "warning",
                "message": "Dataset empty or unprocessed. Baseline model checkpoint saved.",
                "best_model_path": str(best_model_path),
                "last_model_path": str(last_model_path),
            }

        last_val_metrics: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            # 1. Training Phase
            model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            scheduler.step()
            train_loss /= len(train_loader.dataset)

            # 2. Validation Phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            if val_loader and len(val_loader.dataset) > 0:
                with torch.no_grad():
                    for images, targets in val_loader:
                        images, targets = (
                            images.to(self.device),
                            targets.to(self.device),
                        )
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item() * images.size(0)
                        probs = torch.softmax(outputs, dim=1).detach()
                        val_preds.append(probs)
                        val_targets.append(targets)

                val_loss /= len(val_loader.dataset)
                val_preds_cat = torch.cat(val_preds, dim=0)
                val_targets_cat = torch.cat(val_targets, dim=0)
                last_val_metrics = calculate_classification_metrics(
                    val_targets_cat, val_preds_cat, num_classes=self.num_classes
                )
            else:
                val_loss = train_loss
                last_val_metrics = {"accuracy": 0.0, "top_3_accuracy": 0.0, "f1_macro": 0.0}

            # Save best model checkpoint
            if last_val_metrics.get("f1_macro", 0.0) >= best_val_f1:
                best_val_f1 = last_val_metrics.get("f1_macro", 0.0)
                torch.save(model.state_dict(), best_model_path)

            # Check early stopping
            if early_stopping.check(val_loss):
                pass

            if early_stopping.early_stop:
                break

        # Save final checkpoint
        torch.save(model.state_dict(), last_model_path)

        # Log via MLflowTracker
        run_id = self.tracker.log_efficientnet_experiment(
            dataset_version=self.dataset_version,
            hyperparameters={
                "learning_rate": self.learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "seed": self.seed,
                "device": str(self.device),
            },
            evaluation_metrics={
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "accuracy": float(last_val_metrics.get("accuracy", 0.0)),
                "precision_macro": float(last_val_metrics.get("precision_macro", 0.0)),
                "recall_macro": float(last_val_metrics.get("recall_macro", 0.0)),
                "f1_macro": float(last_val_metrics.get("f1_macro", 0.0)),
                "f1_weighted": float(last_val_metrics.get("f1_weighted", 0.0)),
                "top_3_accuracy": float(last_val_metrics.get("top_3_accuracy", 0.0)),
            },
            model_path=best_model_path if best_model_path.exists() else None,
            model_version="1.0.0",
        )

        return {
            "status": "success",
            "epochs_run": epoch,
            "best_val_f1": best_val_f1,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "mlflow_run_id": run_id,
        }
