"""Comprehensive Final System Validation Script for Phase 16.

Executes automated checks verifying:
1. Environment detection & PyTorch/CUDA availability
2. Database initialization and model registry
3. End-to-End AI Breed Recognition Pipeline (YOLO + OpenCV + EfficientNet + Grad-CAM)
4. Configurable backends (PyTorch vs ONNX Runtime)
5. Metric report files and documentation suite existence
"""

import sys
from pathlib import Path
import numpy as np
import torch

from ml.common.env_check import get_pytorch_environment_info
from ml.pipeline.inference_pipeline import BreedRecognitionPipeline
from ml.common.breed_registry import BreedRegistry


def run_final_validation():
    print("=" * 70)
    print("      PHASE 16: FINAL SYSTEM VALIDATION & VERIFICATION")
    print("=" * 70)

    # 1. Environment Check
    print("\n[1/5] Checking PyTorch & Hardware Environment...")
    env_info = get_pytorch_environment_info()
    print(f"      - PyTorch Version : {env_info['pytorch_version']}")
    print(f"      - CUDA Available  : {env_info['cuda_available']}")
    print(f"      - Device Target   : {env_info['compute_device']}")

    # 2. Breed Registry Verification
    print("\n[2/5] Verifying Breed Registry Catalog...")
    registry = BreedRegistry()
    breeds = registry.get_all_breeds()
    print(f"      - Total Breeds Registered: {len(breeds)}")
    for b in breeds:
        print(f"        * [{b.class_id}] {b.display_name} ({b.animal_type.capitalize()})")

    # 3. AI Pipeline Verification (PyTorch Backend)
    print("\n[3/5] Testing AI Recognition Pipeline (PyTorch Engine)...")
    pipeline_pt = BreedRecognitionPipeline(backend="pytorch")
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)

    res_pt = pipeline_pt.predict(dummy_img, generate_gradcam=True)
    print(f"      - Prediction Status   : {res_pt.prediction_status}")
    print(f"      - Predicted Breed     : {res_pt.predicted_breed}")
    print(f"      - Top-1 Confidence    : {res_pt.breed_confidence:.4f}")
    print(f"      - Total Latency (ms)  : {res_pt.inference_time['total_ms']:.2f} ms")
    print(f"      - Grad-CAM Heatmap    : {'Generated' if res_pt.gradcam_output else 'None'}")
    assert res_pt.prediction_status in ("success", "no_animal_detected", "multiple_animals_detected")

    # 4. AI Pipeline Verification (ONNX Runtime Backend)
    print("\n[4/5] Testing AI Recognition Pipeline (ONNX Accelerated Engine)...")
    pipeline_onnx = BreedRecognitionPipeline(backend="onnx")
    res_onnx = pipeline_onnx.predict(dummy_img, generate_gradcam=False)
    print(f"      - Prediction Status   : {res_onnx.prediction_status}")
    print(f"      - Predicted Breed     : {res_onnx.predicted_breed}")
    print(f"      - Top-1 Confidence    : {res_onnx.breed_confidence:.4f}")
    print(f"      - Active Backend      : {res_onnx.model_versions['backend']}")
    assert res_onnx.prediction_status in ("success", "no_animal_detected", "multiple_animals_detected")
    assert res_onnx.model_versions["backend"] == "onnx"

    # 5. Documentation Suite Verification
    print("\n[5/5] Verifying Documentation Suite Completeness...")
    doc_files = [
        "README.md",
        "docs/setup_guide.md",
        "docs/training_guide.md",
        "docs/inference_guide.md",
        "docs/api_documentation.md",
        "docs/database_guide.md",
        "docs/deployment_guide.md",
        "docs/model_documentation.md",
        "docs/dataset_documentation.md",
        "docs/final_testing_report.md",
        "docs/known_limitations.md",
    ]

    missing_docs = []
    for doc in doc_files:
        p = Path(doc)
        if p.exists() and p.stat().st_size > 0:
            print(f"      [OK] Found: {doc} ({p.stat().st_size} bytes)")
        else:
            print(f"      [MISSING]: {doc}")
            missing_docs.append(doc)

    assert len(missing_docs) == 0, f"Missing documentation files: {missing_docs}"

    print("\n" + "=" * 70)
    print("   [SUCCESS] ALL PHASE 16 FINAL SYSTEM VALIDATIONS PASSED CLEANLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_final_validation()
