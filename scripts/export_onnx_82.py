"""
Export trained 82-class EfficientNet-B0 PyTorch model to ONNX format.
Output: models/efficientnet_b0_82_breeds.onnx
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
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import onnx

WORKSPACE = Path(__file__).resolve().parent.parent
MODELS_DIR = WORKSPACE / "models"
PYTORCH_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds_best.pth"
ONNX_MODEL_PATH = MODELS_DIR / "efficientnet_b0_82_breeds.onnx"
CLASS_MAPPING_PATH = MODELS_DIR / "class_names.json"


def export_onnx():
    print("=" * 70)
    print("EXPORTING EFFICIENTNET-B0 (82 BREEDS) TO ONNX")
    print("=" * 70)

    if not PYTORCH_MODEL_PATH.exists():
        print(f"Error: {PYTORCH_MODEL_PATH} not found.")
        sys.exit(1)
    if not CLASS_MAPPING_PATH.exists():
        print(f"Error: {CLASS_MAPPING_PATH} not found.")
        sys.exit(1)

    with open(CLASS_MAPPING_PATH, mode="r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_classes = mapping["num_classes"]

    # 1. Load PyTorch model
    print(f"Loading PyTorch weights: {PYTORCH_MODEL_PATH.name}...")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Export to ONNX
    print(f"Exporting model to ONNX: {ONNX_MODEL_PATH.name}...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )

    # 3. Verify ONNX model integrity
    onnx_model = onnx.load(str(ONNX_MODEL_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model checker verified structural integrity successfully.")
    print(f"Exported ONNX model saved to: {ONNX_MODEL_PATH} ({ONNX_MODEL_PATH.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    export_onnx()
