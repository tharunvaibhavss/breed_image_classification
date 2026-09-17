"""Unit and Integration tests for ONNX model export and ONNX Runtime inference."""

from pathlib import Path
import numpy as np
import pytest
import torch

from ml.classification.efficientnet import BreedClassifier
from ml.export.onnx_exporter import ONNXExporter
from ml.export.onnx_predictor import ONNXBreedPredictor
from ml.pipeline.inference_pipeline import BreedRecognitionPipeline


@pytest.fixture(scope="module")
def exported_onnx_path(tmp_path_factory) -> Path:
    """Fixture exporting PyTorch EfficientNet-B0 to ONNX for testing."""
    tmp_dir = tmp_path_factory.mktemp("onnx_test")
    target_path = tmp_dir / "test_efficientnet_b0.onnx"

    torch.manual_seed(42)
    model = BreedClassifier(num_classes=6)
    model.eval()

    out_path = ONNXExporter.export_efficientnet(
        model=model,
        output_path=str(target_path),
        input_shape=(1, 3, 224, 224),
        dynamic_axes=True,
    )
    return out_path


def test_efficientnet_onnx_export_and_verification(exported_onnx_path: Path):
    """Verify ONNX export creates file and passes structural graph validation."""
    assert exported_onnx_path.exists()
    assert exported_onnx_path.stat().st_size > 0

    is_valid = ONNXExporter.verify_onnx(str(exported_onnx_path))
    assert is_valid is True


def test_onnx_runtime_vs_pytorch_prediction_consistency(exported_onnx_path: Path):
    """Verify PyTorch and ONNX Runtime logits match within 1e-4 tolerance."""
    torch.manual_seed(42)
    model = BreedClassifier(num_classes=6)
    model.eval()

    # Re-export model to guarantee matching weights in test comparison
    ONNXExporter.export_efficientnet(
        model=model,
        output_path=str(exported_onnx_path),
        input_shape=(1, 3, 224, 224),
        dynamic_axes=True,
    )

    dummy_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pytorch_logits = model(dummy_tensor).cpu().numpy()[0]

    predictor = ONNXBreedPredictor(onnx_model_path=str(exported_onnx_path))
    onnx_res = predictor.predict(dummy_tensor)
    onnx_logits = np.array(onnx_res["raw_logits"])

    max_diff = float(np.max(np.abs(pytorch_logits - onnx_logits)))
    assert max_diff < 1e-4, f"PyTorch vs ONNX logit diff {max_diff} exceeded 1e-4 tolerance."


def test_onnx_breed_predictor_top3_format(exported_onnx_path: Path):
    """Verify ONNXBreedPredictor returns valid predicted breed, confidence, and Top-3 list."""
    predictor = ONNXBreedPredictor(onnx_model_path=str(exported_onnx_path))
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

    res = predictor.predict(dummy_np)
    assert "predicted_breed" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_3_predictions"]) == 3
    assert len(res["raw_logits"]) == 6


def test_pipeline_backend_switching(exported_onnx_path: Path):
    """Verify BreedRecognitionPipeline supports switching between 'pytorch' and 'onnx' backends."""
    pipeline_pt = BreedRecognitionPipeline(backend="pytorch")
    assert pipeline_pt.backend == "pytorch"

    pipeline_onnx = BreedRecognitionPipeline(backend="onnx")
    assert pipeline_onnx.backend == "onnx"

    # Test end-to-end inference execution under ONNX backend
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = pipeline_onnx.predict(dummy_img, generate_gradcam=False)
    assert res.model_versions["backend"] == "onnx"
    assert len(res.top_3_predictions) == 3
