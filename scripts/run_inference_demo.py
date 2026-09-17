#!/usr/bin/env python3
"""CLI script to run complete end-to-end AI Breed Recognition Pipeline inference demo."""

import sys
import os
import json
from pathlib import Path
import numpy as np

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.pipeline.inference_pipeline import BreedRecognitionPipeline, InferenceResult


def main():
    """Execute Phase 10 Complete AI Inference Pipeline demo."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    yolo_best = models_dir / "yolo_best.pt"
    eff_best = models_dir / "efficientnet_best.pth"

    print("=" * 70)
    print(" PHASE 10 - COMPLETE AI INFERENCE PIPELINE DEMO")
    print("=" * 70)
    print(f" YOLO Weights Path        : {yolo_best if yolo_best.exists() else 'yolov8n.pt backbone'}")
    print(f" EfficientNet Weights Path: {eff_best}")
    print("-" * 70)

    # 1. Instantiate End-to-End Pipeline
    print("[1/2] Initializing BreedRecognitionPipeline inference service...")
    pipeline = BreedRecognitionPipeline(
        yolo_model_path=yolo_best if yolo_best.exists() else None,
        efficientnet_model_path=eff_best if eff_best.exists() else None,
    )

    # 2. Run Inference on Sample Image Input (400x400 RGB array)
    print("[2/2] Running end-to-end AI inference pipeline...")
    sample_img = np.random.randint(40, 220, (400, 400, 3), dtype=np.uint8)

    result: InferenceResult = pipeline.predict(source=sample_img, generate_gradcam=True)

    print("=" * 70)
    print(" PIPELINE INFERENCE RESPONSE DTO")
    print("=" * 70)
    print(f" Prediction Status   : {result.prediction_status}")
    print(f" Animal Type         : {result.animal_type}")
    print(f" Animal Confidence   : {result.animal_confidence * 100:.2f}%")
    print(f" Bounding Box        : {result.bounding_box}")
    print(f" Predicted Breed     : {result.predicted_breed}")
    print(f" Breed Confidence    : {result.breed_confidence * 100:.2f}%")
    print("-" * 70)
    print(" TOP-3 PREDICTIONS:")
    for rank, p in enumerate(result.top_3_predictions, start=1):
        print(f"   {rank}. {p['display_name']:<20}: {p['confidence'] * 100:.2f}%")
    print("-" * 70)
    print(" INFERENCE LATENCY BREAKDOWN (ms):")
    for k, v in result.inference_time.items():
        print(f"   - {k:<18}: {v:.2f} ms")
    print("-" * 70)
    print(f" Grad-CAM Overlay Generated: {result.gradcam_output is not None}")
    print("=" * 70)


if __name__ == "__main__":
    main()
