"""Albumentations Data Augmentation Pipeline for Breed Classification & YOLO Detection.

Provides realistic, domain-specific training augmentations and deterministic validation/testing
transforms with full YOLO bounding box transformation support.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import albumentations as A


class AugmentationPipeline:
    """Manager constructing Albumentations transform pipelines for training, val, and test."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_to_unit: bool = True,
    ):
        """Initialize AugmentationPipeline.

        Args:
            target_size: Tuple of (height, width). Default (224, 224).
            normalize_to_unit: Scale uint8 pixel values [0, 255] to float32 [0.0, 1.0].
        """
        self.target_height = target_size[0]
        self.target_width = target_size[1]
        self.normalize_to_unit = normalize_to_unit

    def _get_bbox_params(self) -> A.BboxParams:
        """Construct Albumentations YOLO BboxParams configuration."""
        return A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
            clip=True,
        )

    def get_train_transforms(self, enable_bbox: bool = False) -> A.Compose:
        """Construct realistic training augmentation pipeline.

        Includes:
        - Horizontal Flip (p=0.5)
        - ShiftScaleRotate (rotation limit +/-15 deg, scale +/-10%, shift +/-5%)
        - RandomBrightnessContrast (brightness/contrast limit +/-15%)
        - GaussNoise (mild noise, p=0.3)
        - Resize to target (224, 224)
        - Normalization to float32 [0.0, 1.0]

        Args:
            enable_bbox: If True, attaches YOLO BboxParams for bounding box transformations.

        Returns:
            A.Compose albumentations pipeline.
        """
        transforms_list = [
            A.Resize(height=self.target_height, width=self.target_width),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        ]

        if self.normalize_to_unit:
            transforms_list.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0))

        bbox_params = self._get_bbox_params() if enable_bbox else None
        return A.Compose(transforms_list, bbox_params=bbox_params)

    def get_val_transforms(self, enable_bbox: bool = False) -> A.Compose:
        """Construct deterministic validation transform pipeline (Resize + Normalize).

        Args:
            enable_bbox: If True, attaches YOLO BboxParams.

        Returns:
            A.Compose albumentations pipeline.
        """
        transforms_list = [
            A.Resize(height=self.target_height, width=self.target_width),
        ]

        if self.normalize_to_unit:
            transforms_list.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0))

        bbox_params = self._get_bbox_params() if enable_bbox else None
        return A.Compose(transforms_list, bbox_params=bbox_params)

    def get_test_transforms(self, enable_bbox: bool = False) -> A.Compose:
        """Construct deterministic test transform pipeline (identical to validation).

        Args:
            enable_bbox: If True, attaches YOLO BboxParams.

        Returns:
            A.Compose albumentations pipeline.
        """
        return self.get_val_transforms(enable_bbox=enable_bbox)

    def apply_transform(
        self,
        transform: A.Compose,
        image_rgb: np.ndarray,
        bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
        class_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Execute transform pipeline on an image array and optional YOLO bounding boxes.

        Args:
            transform: A.Compose pipeline instance.
            image_rgb: 3-channel uint8 RGB numpy array.
            bboxes: Optional list of YOLO bboxes [(x_center, y_center, width, height), ...].
            class_labels: Optional list of integer class IDs matching bboxes.

        Returns:
            Dict containing 'image' and optionally 'bboxes' and 'class_labels'.
        """
        if bboxes is not None and class_labels is not None:
            augmented = transform(
                image=image_rgb, bboxes=bboxes, class_labels=class_labels
            )
        else:
            augmented = transform(image=image_rgb)

        return augmented
