"""CLI script for ONNX model export, verification, and latency benchmarking."""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
from pathlib import Path
import numpy as np
import torch

from app.core.logging import setup_logging
from ml.classification.efficientnet import BreedClassifier
from ml.export.onnx_exporter import ONNXExporter
from ml.export.onnx_predictor import ONNXBreedPredictor

logger = setup_logging()


def run_export_and_benchmark(num_runs: int = 100):
    """Export models to ONNX and run 100-iteration PyTorch vs ONNX Runtime benchmark."""
    logger.info("=== Phase 15: ONNX Model Export and Optimization Benchmark ===")

    # 1. Instantiate PyTorch EfficientNet-B0 Model
    model = BreedClassifier(num_classes=6)
    model.eval()

    # 2. Export EfficientNet-B0 to ONNX
    onnx_path = Path("models/onnx/efficientnet_b0.onnx")
    ONNXExporter.export_efficientnet(
        model=model,
        output_path=str(onnx_path),
        input_shape=(1, 3, 224, 224),
        dynamic_axes=True,
    )

    # 3. Export YOLO (if weights present)
    yolo_onnx = ONNXExporter.export_yolo(
        yolo_weights_path="models/yolo/best.pt",
        output_dir="models/onnx",
    )

    # 4. Prepare Dummy Image Batch for Benchmarking
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_np = dummy_input.numpy()

    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    onnx_pred = ONNXBreedPredictor(onnx_model_path=str(onnx_path))
    for _ in range(10):
        _ = onnx_pred.predict(dummy_np)

    # 5. Benchmark PyTorch Inference Latency
    pytorch_times = []
    pytorch_logits_list = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            out = model(dummy_input)
            t1 = time.perf_counter()
            pytorch_times.append((t1 - t0) * 1000.0)
            pytorch_logits_list.append(out.cpu().numpy()[0])

    # 6. Benchmark ONNX Runtime Inference Latency
    onnx_times = []
    onnx_logits_list = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = onnx_pred.predict(dummy_np)
        t1 = time.perf_counter()
        onnx_times.append((t1 - t0) * 1000.0)
        onnx_logits_list.append(out["raw_logits"])

    avg_pt_lat = float(np.mean(pytorch_times))
    avg_onnx_lat = float(np.mean(onnx_times))

    pt_fps = float(1000.0 / avg_pt_lat)
    onnx_fps = float(1000.0 / avg_onnx_lat)

    # Calculate Max Logit Absolute Difference
    pt_logit = pytorch_logits_list[0]
    onnx_logit = np.array(onnx_logits_list[0])
    max_abs_diff = float(np.max(np.abs(pt_logit - onnx_logit)))
    is_consistent = max_abs_diff < 1e-4

    # File Sizes
    pt_size_mb = 16.1  # EfficientNet-B0 float32 PyTorch size (~16MB)
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    logger.info("=== Benchmark Results (100 runs) ===")
    logger.info("PyTorch Avg Latency   : %.2f ms (%.1f FPS)", avg_pt_lat, pt_fps)
    logger.info("ONNX Runtime Latency  : %.2f ms (%.1f FPS)", avg_onnx_lat, onnx_fps)
    logger.info("Max Absolute Logit Diff: %.6f (Consistent: %s)", max_abs_diff, is_consistent)
    logger.info("ONNX Model File Size  : %.2f MB", onnx_size_mb)

    # Save JSON Benchmark Results
    results = {
        "num_runs": num_runs,
        "pytorch": {
            "avg_latency_ms": round(avg_pt_lat, 2),
            "fps": round(pt_fps, 1),
            "estimated_size_mb": round(pt_size_mb, 2),
        },
        "onnx_runtime": {
            "avg_latency_ms": round(avg_onnx_lat, 2),
            "fps": round(onnx_fps, 1),
            "file_size_mb": round(onnx_size_mb, 2),
        },
        "consistency": {
            "max_absolute_logit_diff": max_abs_diff,
            "is_numerically_consistent": is_consistent,
        },
    }

    json_out = Path("data/processed/onnx_benchmark_results.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown Documentation
    doc_out = Path("docs/onnx_optimization_report.md")
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    doc_content = f"""# ONNX Model Export and Optimization Report

## Executive Summary
PyTorch `EfficientNet-B0` breed classification model was exported to ONNX format (`opset_version=14`) with dynamic batch dimension support. Structural graph validity was verified using `onnx.checker`. Performance and numerical consistency were benchmarked over **100 runs**.

---

## Performance Comparison (100 Benchmark Runs)

| Metric | PyTorch (CPU) | ONNX Runtime (CPU) | Speedup / Reduction |
| :--- | :--- | :--- | :--- |
| **Avg Latency** | `{avg_pt_lat:.2f} ms` | `{avg_onnx_lat:.2f} ms` | `{(avg_pt_lat / avg_onnx_lat):.2f}x Faster` |
| **Throughput** | `{pt_fps:.1f} FPS` | `{onnx_fps:.1f} FPS` | `+{(onnx_fps - pt_fps):.1f} FPS` |
| **Model Size** | `{pt_size_mb:.2f} MB` | `{onnx_size_mb:.2f} MB` | `{(onnx_size_mb / pt_size_mb * 100):.1f}% Size` |

---

## Numerical Prediction Consistency

- **Max Absolute Logit Delta**: `{max_abs_diff:.6e}`
- **Acceptance Threshold**: `< 1.0e-4`
- **Verification Status**: **{"PASSED (Numerically Identical)" if is_consistent else "FAILED"}**

---

## Production Deployment Recommendation

- **Configurable Backend Parameter**: `INFERENCE_BACKEND` = `"pytorch"` | `"onnx"`.
- **Default Production Setting**: `pytorch` (Maintained for maximum Grad-CAM backward compatibility until full ONNX runtime deployment).
"""
    with open(doc_out, "w", encoding="utf-8") as f:
        f.write(doc_content)

    logger.info("Saved ONNX report to %s and %s", json_out, doc_out)


if __name__ == "__main__":
    run_export_and_benchmark()
