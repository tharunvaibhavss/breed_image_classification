"""Unit tests for MLflow experiment tracking wrapper, naming conventions, and latency benchmarking."""

import pytest
import numpy as np
from pathlib import Path

from ml.common.mlflow_tracker import (
    MLflowTracker,
    YOLO_EXPERIMENT_NAME,
    EFFICIENTNET_EXPERIMENT_NAME,
    benchmark_inference_latency,
)


def test_experiment_naming_conventions():
    """Verify standardized MLflow experiment naming conventions."""
    assert YOLO_EXPERIMENT_NAME == "indian-cattle-buffalo/yolo-animal-detection"
    assert EFFICIENTNET_EXPERIMENT_NAME == "indian-cattle-buffalo/efficientnet-breed-classification"


def test_benchmark_inference_latency_utility():
    """Verify inference latency benchmarking returns positive float in milliseconds."""

    def mock_predict(x):
        return np.sum(x)

    dummy_input = np.ones((50, 50, 3), dtype=np.uint8)
    latency_ms = benchmark_inference_latency(mock_predict, dummy_input, num_warmup=1, num_runs=3)

    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0


def test_mlflow_tracker_logging():
    """Verify MLflowTracker logs YOLO and EfficientNet experiment runs."""
    tracker = MLflowTracker()

    # Log YOLO experiment run
    yolo_run_id = tracker.log_yolo_experiment(
        dataset_version="dataset_v001",
        hyperparameters={"epochs": 10, "batch_size": 8},
        detection_metrics={"mAP50": 90.0},
        inference_time_ms=15.0,
        model_version="1.0.0",
    )
    assert yolo_run_id is not None or tracker.mlflow_available is False

    # Log EfficientNet experiment run
    eff_run_id = tracker.log_efficientnet_experiment(
        dataset_version="dataset_v001",
        hyperparameters={"learning_rate": 0.001, "batch_size": 16, "optimizer": "AdamW"},
        evaluation_metrics={
            "loss": 0.20,
            "accuracy": 95.0,
            "precision_macro": 94.0,
            "recall_macro": 95.0,
            "f1_macro": 94.5,
            "top_3_accuracy": 99.0,
        },
        inference_time_ms=25.0,
        model_version="1.0.0",
    )
    assert eff_run_id is not None or tracker.mlflow_available is False
