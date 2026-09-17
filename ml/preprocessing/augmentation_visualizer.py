"""Augmentation Visualizer module for rendering side-by-side comparison plots with bounding box overlays.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from ml.preprocessing.albumentations_pipeline import AugmentationPipeline


def draw_yolo_bboxes_on_image(
    image_rgb: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    class_labels: Optional[List[int]] = None,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Draw YOLO normalized bounding boxes on an RGB image array.

    Args:
        image_rgb: 3-channel uint8 or float32 RGB image array.
        bboxes: List of YOLO normalized bboxes [(x_center, y_center, width, height), ...].
        class_labels: List of class IDs corresponding to bboxes.
        color_map: Dict mapping class_id -> RGB color tuple.

    Returns:
        Image copy with bounding boxes drawn.
    """
    img_copy = image_rgb.copy()
    if img_copy.dtype != np.uint8:
        img_copy = np.clip(img_copy * 255.0 if img_copy.max() <= 1.0 else img_copy, 0, 255).astype(np.uint8)

    h, w = img_copy.shape[:2]
    if color_map is None:
        color_map = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green for cattle, Red for buffalo

    for idx, (xc, yc, bw, bh) in enumerate(bboxes):
        cls_id = class_labels[idx] if class_labels and idx < len(class_labels) else 0
        color = color_map.get(cls_id, (0, 255, 0))

        # Convert normalized YOLO (x_center, y_center, w, h) to pixel (xmin, ymin, xmax, ymax)
        xmin = int(round((xc - bw / 2.0) * w))
        ymin = int(round((yc - bh / 2.0) * h))
        xmax = int(round((xc + bw / 2.0) * w))
        ymax = int(round((yc + bh / 2.0) * h))

        # Clamp to image bounds
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w - 1, xmax), min(h - 1, ymax)

        # Draw rectangle box
        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, 2)

        # Label text
        label_text = "Cattle" if cls_id == 0 else "Buffalo"
        cv2.putText(
            img_copy,
            label_text,
            (xmin, max(15, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return img_copy


def generate_augmentation_comparison_plot(
    image_rgb: np.ndarray,
    bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    class_labels: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate side-by-side comparison plot of Original vs. Augmented image.

    Args:
        image_rgb: uint8 RGB numpy array.
        bboxes: Optional YOLO bounding boxes.
        class_labels: Optional class IDs matching bboxes.
        output_path: Path to save generated plot PNG file.

    Returns:
        Path to output plot PNG.
    """
    if output_path is None:
        output_path = Path("docs/augmentation_visualization.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize augmentation pipeline
    pipeline = AugmentationPipeline(target_size=(224, 224), normalize_to_unit=False)
    enable_bbox = bboxes is not None and len(bboxes) > 0
    transform = pipeline.get_train_transforms(enable_bbox=enable_bbox)

    # Apply transform
    augmented = pipeline.apply_transform(
        transform, image_rgb, bboxes=bboxes, class_labels=class_labels
    )
    aug_img = augmented["image"]
    aug_bboxes = augmented.get("bboxes", [])
    aug_labels = augmented.get("class_labels", [])

    # Render bounding box overlays
    orig_drawn = (
        draw_yolo_bboxes_on_image(image_rgb, bboxes, class_labels)
        if bboxes
        else image_rgb
    )
    aug_drawn = (
        draw_yolo_bboxes_on_image(aug_img, aug_bboxes, aug_labels)
        if aug_bboxes
        else aug_img
    )

    # Create figure plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(orig_drawn)
    axes[0].set_title("Original Image (Unaugmented)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(aug_drawn)
    axes[1].set_title("Albumentations Augmented Image", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path
