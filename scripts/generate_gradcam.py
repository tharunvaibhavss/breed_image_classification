#!/usr/bin/env python3
"""CLI script to generate Grad-CAM explainability visual heatmap overlay for sample breed image."""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.explainability.gradcam import EfficientNetGradCAM
from ml.explainability.visualizer import GradCAMVisualizer


def main():
    """Execute Phase 8 Grad-CAM explainability generation demo."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    docs_dir = project_root / "docs"

    best_model_path = models_dir / "efficientnet_best.pth"
    output_plot_path = docs_dir / "gradcam_visualization.png"

    print("=" * 70)
    print(" PHASE 8 - GRAD-CAM EXPLAINABILITY GENERATION")
    print("=" * 70)
    print(f" Model Weights  : {best_model_path}")
    print(f" Output Plot    : {output_plot_path}")
    print("-" * 70)

    # Generate synthetic 400x400 RGB sample image for demonstration
    dummy_sample = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)

    # 1. Instantiate Grad-CAM engine
    print("[1/2] Initializing EfficientNet-B0 Grad-CAM engine...")
    gradcam_engine = EfficientNetGradCAM(model_path=best_model_path)

    # 2. Generate Grad-CAM heatmap and overlay
    print("[2/2] Computing class activation map and generating visual overlay...")
    result = gradcam_engine.generate_gradcam(source=dummy_sample, alpha=0.5)

    # 3. Render side-by-side comparison plot
    visualizer = GradCAMVisualizer()
    plot_file = visualizer.plot_gradcam_comparison(result, output_path=output_plot_path)

    print("=" * 70)
    print(" GRAD-CAM EXPLAINABILITY SUMMARY")
    print("=" * 70)
    print(f" Predicted Breed    : {result.display_name} ({result.predicted_breed})")
    print(f" Animal Type        : {result.animal_type}")
    print(f" Confidence Score   : {result.confidence * 100:.2f}%")
    print(f" Target Class ID    : {result.target_class_id}")
    print(f" Heatmap Dimensions : {result.heatmap.shape}")
    print(f" Overlay Dimensions : {result.overlay_image.shape}")
    print(f" Visual Plot Saved  : {plot_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
