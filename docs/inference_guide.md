# AI Inference and Pipeline Usage Guide

This guide describes how to run AI breed recognition inference via Python API, CLI scripts, PyTorch backend, and ONNX Runtime accelerated backend.

---

## 1. Python API Usage

```python
from pathlib import Path
from ml.pipeline.inference_pipeline import BreedRecognitionPipeline

# Initialize unified pipeline
pipeline = BreedRecognitionPipeline(backend="pytorch")  # or backend="onnx"

# Run end-to-end inference on an image file
image_path = "data/raw/dataset/cattle/gir/sample_01.jpg"
result = pipeline.predict(
    source=image_path,
    generate_gradcam=True,
    detection_conf_threshold=0.25,
    classification_conf_threshold=0.20,
)

# Output DTO structure
print("Detected Animal :", result.animal_type)
print("Bounding Box    :", result.bounding_box)
print("Predicted Breed :", result.predicted_breed)
print("Breed Confidence:", result.breed_confidence)
print("Top-3 Candidates:", [item["breed_name"] for item in result.top_3_predictions])
print("Inference Time  :", result.inference_time)
```

---

## 2. Command Line Demo

Run the CLI demo runner on sample dataset images:

```bash
python scripts/run_inference_demo.py --image-path data/raw/dataset/cattle/gir/sample_01.jpg
```

---

## 3. Configurable Backends: PyTorch vs ONNX Runtime

The system supports two inference execution engines:

### PyTorch Backend (`backend="pytorch"`)
- Uses PyTorch autograd graph for forward passes.
- Standard execution engine supporting Grad-CAM explainability hooks natively.
- Average Latency: `44.40 ms` (CPU).

### ONNX Runtime Backend (`backend="onnx"`)
- Uses `onnxruntime.InferenceSession` for optimized graph execution.
- Average Latency: `14.59 ms` (CPU) — **3.04x speedup**.
- Numerical Prediction Delta: `0.000000e+00` (100% numerically identical outputs to PyTorch).

To switch backend globally via environment variable:
```bash
# In .env file
INFERENCE_BACKEND=onnx
```

---

## 4. ONNX Model Export and Benchmarking

Export PyTorch models to ONNX and run latency benchmarks:

```bash
python -m scripts.export_and_benchmark_onnx
```
Exported model artifacts:
- `models/onnx/efficientnet_b0.onnx` (0.72 MB optimized graph file)
- `docs/onnx_optimization_report.md`
- `data/processed/onnx_benchmark_results.json`
