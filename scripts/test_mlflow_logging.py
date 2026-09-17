#!/usr/bin/env python3
"""CLI script to test and verify MLflow experiment logging for YOLO and EfficientNet-B0."""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.common.mlflow_tracker import (
    MLflowTracker,
    YOLO_EXPERIMENT_NAME,
    EFFICIENTNET_EXPERIMENT_NAME,
    benchmark_inference_latency,
)
from ml.classification.predictor import BreedPredictor


def main():
    """Execute MLflow logging and inference latency benchmark verification."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    best_model = models_dir / "efficientnet_best.pth"

    print("=" * 70)
    print(" PHASE 9 - MLFLOW EXPERIMENT MANAGEMENT & LOGGING TEST")
    print("=" * 70)
    print(f" YOLO Experiment Name         : {YOLO_EXPERIMENT_NAME}")
    print(f" EfficientNet Experiment Name : {EFFICIENTNET_EXPERIMENT_NAME}")
    print("-" * 70)

    tracker = MLflowTracker()

    # 1. Benchmark Model Inference Latency
    print("[1/3] Benchmarking EfficientNet-B0 model inference latency...")
    predictor = BreedPredictor()
    dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    latency_ms = benchmark_inference_latency(
        predict_fn=lambda img: predictor.predict(img),
        sample_input=dummy_input,
        num_warmup=2,
        num_runs=5,
    )
    print(f" Average Inference Latency: {latency_ms:.2f} ms per sample")

    # 2. Test Logging YOLO Detection Experiment Run
    print("[2/3] Logging YOLO Animal Detection Experiment Run...")
    yolo_run_id = tracker.log_yolo_experiment(
        dataset_version="dataset_v001",
        hyperparameters={
            "weights_name": "yolov8n.pt",
            "epochs": 50,
            "batch_size": 16,
            "img_size": 640,
            "device": "cpu",
        },
        detection_metrics={
            "precision": 92.5,
            "recall": 89.1,
            "mAP50": 94.2,
            "mAP50_95": 78.6,
        },
        model_path=models_dir / "yolo_best.pt" if (models_dir / "yolo_best.pt").exists() else None,
        inference_time_ms=12.5,
        model_version="1.0.0",
        run_name="yolo-animal-detection-test-run",
    )

    # 3. Test Logging EfficientNet Breed Classification Experiment Run
    print("[3/3] Logging EfficientNet-B0 Breed Classification Experiment Run...")
    eff_run_id = tracker.log_efficientnet_experiment(
        dataset_version="dataset_v001",
        hyperparameters={
            "learning_rate": 1e-3,
            "batch_size": 16,
            "epochs": 20,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "seed": 42,
            "device": "cpu",
        },
        evaluation_metrics={
            "loss": 0.154,
            "accuracy": 95.2,
            "precision_macro": 94.8,
            "recall_macro": 95.0,
            "f1_macro": 94.9,
            "f1_weighted": 95.1,
            "top_3_accuracy": 99.1,
        },
        model_path=best_model if best_model.exists() else None,
        inference_time_ms=latency_ms,
        model_version="1.0.0",
        run_name="efficientnet-b0-classification-test-run",
    )

    print("=" * 70)
    print(" MLFLOW LOGGING SUMMARY")
    print("=" * 70)
    print(f" MLflow Tracking URI          : {tracker.tracking_uri}")
    print(f" YOLO Run ID                 : {yolo_run_id}")
    print(f" EfficientNet Run ID         : {eff_run_id}")
    print(f" Measured Inference Latency  : {latency_ms:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
