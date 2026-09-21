"""
Export Model v3 to ONNX and Benchmark PyTorch vs. ONNX Runtime Latency.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import onnx
import onnxruntime as ort

WORKSPACE = Path(__file__).resolve().parent.parent
MODELS_V3 = WORKSPACE / "models" / "synthetic_82_breeds_v3"
REPORTS_V3 = WORKSPACE / "reports" / "synthetic_v3"
REPORTS_V2 = WORKSPACE / "reports" / "dataset_expansion_v2"
SPLIT_MANIFEST_REAL = REPORTS_V2 / "split_manifest_v2.csv"

PYTORCH_MODEL_PATH = MODELS_V3 / "best_model_v3.pth"
ONNX_MODEL_PATH = MODELS_V3 / "efficientnet_b0_v3.onnx"
CLASS_MAPPING_PATH = MODELS_V3 / "class_mapping_v3.json"


def main():
    print("=" * 70)
    print("EXPORTING MODEL v3 TO ONNX & BENCHMARKING LATENCY")
    print("=" * 70)

    with open(CLASS_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    num_classes = mapping["num_classes"]

    # 1. Load PyTorch model
    print("Loading PyTorch model...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    ckpt = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Export to ONNX
    print(f"Exporting to {ONNX_MODEL_PATH}...")
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Exported successfully.")

    # 3. Verify ONNX model
    print("Verifying ONNX model graph...")
    onnx_model = onnx.load(str(ONNX_MODEL_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified.")

    # 4. Parity check between PyTorch and ONNX Runtime
    print("Checking numerical parity...")
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    
    test_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = model(test_tensor).numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: test_tensor.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Maximum absolute difference between PyTorch and ONNX: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Discrepancy too high: {max_diff}"
    print("Numerical parity check PASSED (< 1e-4).")

    # 5. Benchmark Latency
    print("\nBenchmarking latency (50 iterations)...")
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor_input = eval_tf(dummy_img).unsqueeze(0)
    numpy_input = tensor_input.numpy()

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(tensor_input)
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: numpy_input})

    # PyTorch timing
    pt_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor_input)
        pt_times.append((time.perf_counter() - t0) * 1000)

    # ONNX timing
    ort_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: numpy_input})
        ort_times.append((time.perf_counter() - t0) * 1000)

    pt_mean = np.mean(pt_times)
    ort_mean = np.mean(ort_times)
    speedup = pt_mean / ort_mean

    print(f"PyTorch CPU Latency:  {pt_mean:.2f} ms (+/- {np.std(pt_times):.2f} ms)")
    print(f"ONNX Runtime Latency: {ort_mean:.2f} ms (+/- {np.std(ort_times):.2f} ms)")
    print(f"ONNX Speedup Factor:  {speedup:.2f}x")

    # Save benchmark results
    bench_data = {
        "model": "efficientnet_b0_v3",
        "pytorch_latency_mean_ms": round(pt_mean, 2),
        "pytorch_latency_std_ms": round(float(np.std(pt_times)), 2),
        "onnx_latency_mean_ms": round(ort_mean, 2),
        "onnx_latency_std_ms": round(float(np.std(ort_times)), 2),
        "speedup_factor": round(speedup, 2),
        "max_numerical_diff": float(max_diff),
        "parity_passed": bool(max_diff < 1e-4)
    }

    with open(MODELS_V3 / "onnx_benchmark_v3.json", "w") as f:
        json.dump(bench_data, f, indent=2)
    print(f"Benchmark saved to: {MODELS_V3 / 'onnx_benchmark_v3.json'}")


if __name__ == "__main__":
    main()
