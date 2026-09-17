"""YOLO Detection Model Trainer module with MLflow experiment tracking integration.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config import settings
from ml.common.mlflow_tracker import MLflowTracker, YOLO_EXPERIMENT_NAME


class YOLOModelTrainer:
    """Trainer manager orchestrating lightweight YOLO animal detection training."""

    def __init__(
        self,
        config_path: Path,
        weights_name: str = "yolov8n.pt",
        output_dir: Optional[Path] = None,
        dataset_version: str = "dataset_v001",
    ):
        """Initialize YOLOModelTrainer.

        Args:
            config_path: Path to dataset yolo_config.yaml file.
            weights_name: Base model weights name (default 'yolov8n.pt').
            output_dir: Directory path to save output checkpoints.
            dataset_version: Dataset release version string.
        """
        self.config_path = Path(config_path)
        self.weights_name = weights_name
        self.output_dir = Path(output_dir) if output_dir else Path("models")
        self.dataset_version = dataset_version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = MLflowTracker()

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Execute YOLO detection training with MLflow tracking.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size.
            img_size: Input resolution size (default 640).
            device: Compute device ('cpu' or 'cuda:0').

        Returns:
            Dict containing training result metrics and saved model paths.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"YOLO configuration file not found at: {self.config_path}")

        try:
            from ultralytics import YOLO

            model = YOLO(self.weights_name)
            results = model.train(
                data=str(self.config_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=device,
                project=str(self.output_dir / "yolo_runs"),
                name="train_exp",
                save=True,
                exist_ok=True,
                verbose=True,
            )

            # Save best and last models to target models directory
            best_source = self.output_dir / "yolo_runs" / "train_exp" / "weights" / "best.pt"
            last_source = self.output_dir / "yolo_runs" / "train_exp" / "weights" / "last.pt"

            best_target = self.output_dir / "yolo_best.pt"
            last_target = self.output_dir / "yolo_last.pt"

            if best_source.exists():
                import shutil
                shutil.copy(best_source, best_target)

            if last_source.exists():
                import shutil
                shutil.copy(last_source, last_target)

            # Log experiment via MLflowTracker
            run_id = self.tracker.log_yolo_experiment(
                dataset_version=self.dataset_version,
                hyperparameters={
                    "weights_name": self.weights_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "img_size": img_size,
                    "device": device,
                },
                detection_metrics={
                    "epochs_completed": float(epochs),
                },
                model_path=best_target if best_target.exists() else None,
                model_version="1.0.0",
            )

            return {
                "status": "success",
                "epochs_completed": epochs,
                "best_model_path": str(best_target),
                "last_model_path": str(last_target),
                "mlflow_run_id": run_id,
            }
        except ImportError as e:
            raise ImportError(f"Required package missing for YOLO training: {str(e)}")
