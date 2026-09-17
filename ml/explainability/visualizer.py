"""Grad-CAM Visualizer module for rendering side-by-side explainability plots.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from ml.explainability.gradcam import GradCAMResult


class GradCAMVisualizer:
    """Visualizer rendering comparison plots of original image, heatmap, and overlay."""

    def plot_gradcam_comparison(
        self,
        result: GradCAMResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Render 3-panel side-by-side Grad-CAM explanation figure.

        Args:
            result: GradCAMResult dataclass instance.
            output_path: Target PNG output path (default docs/gradcam_visualization.png).

        Returns:
            Path to saved plot file.
        """
        if output_path is None:
            output_path = Path("docs/gradcam_visualization.png")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original Image
        axes[0].imshow(result.original_image)
        axes[0].set_title("Original Input Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM Heatmap
        im2 = axes[1].imshow(result.heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Blended Overlay
        axes[2].imshow(result.overlay_image)
        axes[2].set_title("Blended Explanation Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(
            f"EfficientNet-B0 Grad-CAM Explanation\n"
            f"Predicted: {result.display_name} ({result.predicted_breed}) | "
            f"Confidence: {result.confidence * 100:.2f}%",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path
