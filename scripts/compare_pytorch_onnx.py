"""
Compare PyTorch vs ONNX Runtime inference on the unseen test dataset (82 Breeds).
Verifies:
  - Prediction agreement rate
  - Max and mean probability difference
  - Latency comparison (Mean, Median, P95, P99)
  - Speedup factor
Outputs:
  - reports/onnx_validation_report.json
  - reports/onnx_validation_report.md
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
import onnxruntime as ort
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_ROOT = WORKSPACE / "dataset"
SPLITS_DIR = DATASET_ROOT / "splits"
MODELS_DIR = WORKSPACE / "models"
REPORTS_DIR = WORKSPACE / "reports"

TEST_CSV = SPLITS_DIR / "test.csv"
CLASS_MAPPING_PATH = MODELS_DIR / "class_names.json"
PYTORCH_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds_best.pth"
ONNX_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds.onnx"
REPORT_JSON = REPORTS_DIR / "onnx_validation_report.json"
REPORT_MD = REPORTS_DIR / "onnx_validation_report.md"


def compare_pytorch_onnx():
    print("=" * 70)
    print("BENCHMARKING PYTORCH VS ONNX RUNTIME (82 BREEDS)")
    print("=" * 70)

    if not PYTORCH_MODEL_PATH.exists() or not ONNX_MODEL_PATH.exists():
        print(f"Error: Model files not found. Ensure models are trained and exported.")
        sys.exit(1)

    with open(CLASS_MAPPING_PATH, mode="r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_classes = mapping["num_classes"]

    # Load PyTorch model
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load ONNX Runtime
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), ort_options, providers=["CPUExecutionProvider"])
    ort_input_name = ort_session.get_inputs()[0].name

    # Transform
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_samples = []
    with open(TEST_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fp = DATASET_ROOT / r["relative_path"]
            if fp.exists():
                test_samples.append(fp)

    print(f"Loaded {len(test_samples)} test samples for parity benchmark.")

    agreements = 0
    max_prob_diff = 0.0
    prob_diffs = []
    pytorch_times = []
    onnx_times = []

    # Warmup
    dummy_input = torch.randn(1, 3, 224, 224)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
        _ = ort_session.run(None, {ort_input_name: dummy_input.numpy()})

    for fp in test_samples:
        img = Image.open(fp).convert("RGB")
        tensor = eval_transform(img).unsqueeze(0)
        np_input = tensor.numpy()

        # PyTorch
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_logits = model(tensor)
            pt_probs = torch.softmax(pt_logits, dim=1).squeeze(0).numpy()
        pt_time = (time.perf_counter() - t0) * 1000.0
        pytorch_times.append(pt_time)

        # ONNX
        t0 = time.perf_counter()
        ort_outs = ort_session.run(None, {ort_input_name: np_input})
        onnx_logits = ort_outs[0][0]
        exp_l = np.exp(onnx_logits - np.max(onnx_logits))
        onnx_probs = exp_l / np.sum(exp_l)
        ort_time = (time.perf_counter() - t0) * 1000.0
        onnx_times.append(ort_time)

        pt_pred = int(np.argmax(pt_probs))
        onnx_pred = int(np.argmax(onnx_probs))

        if pt_pred == onnx_pred:
            agreements += 1

        diff = np.abs(pt_probs - onnx_probs)
        max_diff = float(np.max(diff))
        if max_diff > max_prob_diff:
            max_prob_diff = max_diff
        prob_diffs.append(float(np.mean(diff)))

    total = len(test_samples)
    parity_rate = (agreements / total) * 100.0 if total > 0 else 0.0
    mean_diff = float(np.mean(prob_diffs))

    pt_mean = float(np.mean(pytorch_times))
    pt_median = float(np.median(pytorch_times))
    pt_p95 = float(np.percentile(pytorch_times, 95))
    pt_p99 = float(np.percentile(pytorch_times, 99))

    ort_mean = float(np.mean(onnx_times))
    ort_median = float(np.median(onnx_times))
    ort_p95 = float(np.percentile(onnx_times, 95))
    ort_p99 = float(np.percentile(onnx_times, 99))

    speedup = (pt_mean / ort_mean) if ort_mean > 0 else 1.0
    pt_size_mb = PYTORCH_MODEL_PATH.stat().st_size / (1024 * 1024)
    onnx_size_mb = ONNX_MODEL_PATH.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 50)
    print("PYTORCH VS ONNX BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Agreement Rate: {parity_rate:.2f}% ({agreements}/{total})")
    print(f"Max Prob Diff:  {max_prob_diff:.6e}")
    print(f"Mean Prob Diff: {mean_diff:.6e}")
    print(f"PyTorch Mean:   {pt_mean:.2f} ms (Median: {pt_median:.2f} ms)")
    print(f"ONNX Mean:      {ort_mean:.2f} ms (Median: {ort_median:.2f} ms)")
    print(f"Speedup:        {speedup:.2f}x")

    report_data = {
        "status": "PASSED" if parity_rate >= 99.0 and max_prob_diff < 1e-3 else "FAILED",
        "total_test_samples": total,
        "prediction_agreement_count": agreements,
        "prediction_agreement_rate_pct": round(parity_rate, 2),
        "max_probability_difference": float(f"{max_prob_diff:.6e}"),
        "mean_probability_difference": float(f"{mean_diff:.6e}"),
        "pytorch_latency_ms": {
            "mean": round(pt_mean, 2),
            "median": round(pt_median, 2),
            "p95": round(pt_p95, 2),
            "p99": round(pt_p99, 2)
        },
        "onnx_latency_ms": {
            "mean": round(ort_mean, 2),
            "median": round(ort_median, 2),
            "p95": round(ort_p95, 2),
            "p99": round(ort_p99, 2)
        },
        "speedup_factor": round(speedup, 2),
        "model_size_mb": {
            "pytorch": round(pt_size_mb, 2),
            "onnx": round(onnx_size_mb, 2)
        }
    }

    with open(REPORT_JSON, mode="w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    report_md = f"""# ONNX Export and Benchmark Validation Report (82 Breeds)

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model**: EfficientNet-B0 (82 Classes)  
**Status**: **{report_data['status']}**  

---

## 1. Parity and Numerical Verification

| Metric | PyTorch Inference | ONNX Runtime | Parity / Difference |
|---|---|---|---|
| **Top-1 Class Agreement** | - | - | **{parity_rate:.2f}%** ({agreements}/{total} samples) |
| **Max Absolute Diff (Softmax)** | - | - | **{max_prob_diff:.6e}** |
| **Mean Absolute Diff (Softmax)**| - | - | **{mean_diff:.6e}** |
| **Model Disk Size** | {pt_size_mb:.2f} MB | {onnx_size_mb:.2f} MB | {((onnx_size_mb - pt_size_mb)/pt_size_mb)*100:+.1f}% |

---

## 2. Latency Benchmarks (CPU)

| Metric | PyTorch (CPU) | ONNX Runtime (CPU) | Speedup |
|---|---|---|---|
| **Mean Latency** | {pt_mean:.2f} ms | {ort_mean:.2f} ms | **{speedup:.2f}x** |
| **Median Latency** | {pt_median:.2f} ms | {ort_median:.2f} ms | {pt_median / ort_median:.2f}x |
| **95th Percentile (P95)** | {pt_p95:.2f} ms | {ort_p95:.2f} ms | {pt_p95 / ort_p95:.2f}x |
| **99th Percentile (P99)** | {pt_p99:.2f} ms | {ort_p99:.2f} ms | {pt_p99 / ort_p99:.2f}x |
"""
    with open(REPORT_MD, mode="w", encoding="utf-8") as f:
        f.write(report_md)


if __name__ == "__main__":
    compare_pytorch_onnx()
