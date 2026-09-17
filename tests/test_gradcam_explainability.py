"""Unit tests for Grad-CAM explainability engine, heatmap normalization, overlay generation, and model state preservation."""

import numpy as np
import pytest
import torch
from pathlib import Path

from ml.explainability.gradcam import EfficientNetGradCAM, GradCAMResult
from ml.explainability.visualizer import GradCAMVisualizer


def test_gradcam_dimensions_and_types():
    """Verify Grad-CAM output heatmap and overlay match input image dimensions (H, W)."""
    gradcam = EfficientNetGradCAM()
    img_h, img_w = 350, 450
    dummy_img = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)

    res = gradcam.generate_gradcam(dummy_img, alpha=0.5)

    assert isinstance(res, GradCAMResult)
    assert res.original_image.shape == (img_h, img_w, 3)
    assert res.heatmap.shape == (img_h, img_w)
    assert res.overlay_image.shape == (img_h, img_w, 3)
    assert res.overlay_image.dtype == np.uint8


def test_gradcam_heatmap_normalization():
    """Verify heatmap values are normalized in range [0.0, 1.0]."""
    gradcam = EfficientNetGradCAM()
    dummy_img = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)

    res = gradcam.generate_gradcam(dummy_img, alpha=0.5)

    assert np.min(res.heatmap) >= 0.0
    assert np.max(res.heatmap) <= 1.0
    assert res.heatmap.dtype == np.float32


def test_model_prediction_unchanged_after_gradcam():
    """Verify model predictions and weight parameters remain unchanged before and after Grad-CAM."""
    gradcam = EfficientNetGradCAM()
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 1. Forward pass before Grad-CAM
    tensor_img = gradcam.preprocessor.preprocess_to_tensor(dummy_img).unsqueeze(0).to(gradcam.device)
    with torch.no_grad():
        logits_before = gradcam.model(tensor_img).cpu().numpy()

    # Capture weight parameter checksum before Grad-CAM
    params_before = [p.clone() for p in gradcam.model.parameters()]

    # 2. Run Grad-CAM execution
    res = gradcam.generate_gradcam(dummy_img)

    # 3. Forward pass after Grad-CAM
    with torch.no_grad():
        logits_after = gradcam.model(tensor_img).cpu().numpy()

    # Capture weight parameter checksum after Grad-CAM
    params_after = [p.clone() for p in gradcam.model.parameters()]

    # Verify logits and parameters are 100% identical
    np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5, atol=1e-5)
    for p1, p2 in zip(params_before, params_after):
        assert torch.equal(p1, p2), "Model parameters were mutated during Grad-CAM generation!"


def test_gradcam_visualizer_plot_saving(tmp_path: Path):
    """Verify GradCAMVisualizer saves comparison plot."""
    gradcam = EfficientNetGradCAM()
    dummy_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    res = gradcam.generate_gradcam(dummy_img)

    visualizer = GradCAMVisualizer()
    plot_file = tmp_path / "test_gradcam.png"
    out_path = visualizer.plot_gradcam_comparison(res, output_path=plot_file)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
