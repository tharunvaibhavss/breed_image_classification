"""Standardized MLflow Experiment Tracking module for YOLO Detection and EfficientNet-B0 Classification.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from app.core.config import settings

# Standardized MLflow Experiment Naming Conventions
YOLO_EXPERIMENT_NAME = "indian-cattle-buffalo/yolo-animal-detection"
EFFICIENTNET_EXPERIMENT_NAME = "indian-cattle-buffalo/efficientnet-breed-classification"


def benchmark_inference_latency(
    predict_fn: Callable[[Any], Any],
    sample_input: Any,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> float:
    """Benchmark mean model inference latency in milliseconds (ms).

    Args:
        predict_fn: Callable prediction function taking sample_input.
        sample_input: Sample image input array or tensor.
        num_warmup: Number of warmup runs.
        num_runs: Number of timed iteration runs.

    Returns:
        Mean inference latency in milliseconds (ms).
    """
    for _ in range(num_warmup):
        _ = predict_fn(sample_input)

    latencies: list[float] = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = predict_fn(sample_input)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

    return round(float(sum(latencies) / len(latencies)), 2)


class MLflowTracker:
    """Tracker manager enforcing consistent MLflow logging schemas across detection and classification."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflowTracker."""
        import os
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        requested_uri = tracking_uri if tracking_uri else settings.MLFLOW_TRACKING_URI
        # Default to local file-based tracking store if local MLflow server is offline
        if not tracking_uri or requested_uri == "http://localhost:5000":
            requested_uri = "file:./mlruns"

        self.tracking_uri = requested_uri
        self.mlflow_available = False

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self.mlflow_available = True
        except ImportError:
            self.mlflow_available = False

    def log_yolo_experiment(
        self,
        dataset_version: str,
        hyperparameters: Dict[str, Any],
        detection_metrics: Dict[str, float],
        model_path: Optional[Path] = None,
        inference_time_ms: Optional[float] = None,
        model_version: str = "1.0.0",
        run_name: str = "yolo-detection-run",
    ) -> Optional[str]:
        """Log YOLO animal detection experiment parameters, metrics, artifacts, and latency to MLflow.

        Args:
            dataset_version: Version string (e.g. 'dataset_v001').
            hyperparameters: Dict of training hyperparameters (epochs, batch_size, imgsz, etc.).
            detection_metrics: Dict of detection metrics (precision, recall, mAP50, etc.).
            model_path: Path to best model checkpoint file.
            inference_time_ms: Inference latency in milliseconds (ms).
            model_version: Model semantic version string.
            run_name: MLflow run display name.

        Returns:
            MLflow run ID string if logged successfully, else None.
        """
        if not self.mlflow_available:
            return None

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(YOLO_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=run_name) as run:
                # 1. Log System & Dataset Tags
                mlflow.set_tags(
                    {
                        "task": "animal-detection",
                        "framework": "ultralytics-yolo",
                        "dataset_version": dataset_version,
                        "model_version": model_version,
                    }
                )

                # 2. Log Hyperparameters
                params_to_log = {
                    "dataset_version": dataset_version,
                    "model_version": model_version,
                    **hyperparameters,
                }
                mlflow.log_params(params_to_log)

                # 3. Log Metrics & Latency
                metrics_to_log = {**detection_metrics}
                if inference_time_ms is not None:
                    metrics_to_log["inference_time_ms"] = inference_time_ms
                mlflow.log_metrics(metrics_to_log)

                # 4. Log Model Checkpoint Artifact
                if model_path and Path(model_path).exists():
                    mlflow.log_artifact(str(model_path))

                return run.info.run_id
        except Exception as e:
            print(f"[MLflow Error]: {type(e).__name__}: {str(e)}")
            return None

    def log_efficientnet_experiment(
        self,
        dataset_version: str,
        hyperparameters: Dict[str, Any],
        evaluation_metrics: Dict[str, float],
        model_path: Optional[Path] = None,
        inference_time_ms: Optional[float] = None,
        model_version: str = "1.0.0",
        run_name: str = "efficientnet-b0-classification-run",
    ) -> Optional[str]:
        """Log EfficientNet-B0 breed classification experiment parameters, metrics, artifacts, and latency.

        Args:
            dataset_version: Version string (e.g. 'dataset_v001').
            hyperparameters: Dict of training hyperparameters (learning_rate, batch_size, epochs, optimizer, etc.).
            evaluation_metrics: Dict of classification metrics (loss, accuracy, precision, recall, f1, top_3_accuracy, etc.).
            model_path: Path to best model checkpoint file.
            inference_time_ms: Inference latency in milliseconds (ms).
            model_version: Model semantic version string.
            run_name: MLflow run display name.

        Returns:
            MLflow run ID string if logged successfully, else None.
        """
        if not self.mlflow_available:
            return None

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(EFFICIENTNET_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=run_name) as run:
                # 1. Log System & Dataset Tags
                mlflow.set_tags(
                    {
                        "task": "breed-classification",
                        "architecture": "EfficientNet-B0",
                        "dataset_version": dataset_version,
                        "model_version": model_version,
                    }
                )

                # 2. Log Hyperparameters
                params_to_log = {
                    "dataset_version": dataset_version,
                    "model_version": model_version,
                    "architecture": "EfficientNet-B0",
                    **hyperparameters,
                }
                mlflow.log_params(params_to_log)

                # 3. Log Evaluation Metrics & Latency
                metrics_to_log = {**evaluation_metrics}
                if inference_time_ms is not None:
                    metrics_to_log["inference_time_ms"] = inference_time_ms
                mlflow.log_metrics(metrics_to_log)

                # 4. Log Model Checkpoint Artifact
                if model_path and Path(model_path).exists():
                    mlflow.log_artifact(str(model_path))

                return run.info.run_id
        except Exception as e:
            print(f"[MLflow Error]: {type(e).__name__}: {str(e)}")
            return None
