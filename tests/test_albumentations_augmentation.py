"""Unit tests for Albumentations data augmentation pipeline and YOLO bounding box transformations."""

import numpy as np
import pytest
import albumentations as A

from ml.preprocessing.albumentations_pipeline import AugmentationPipeline


def test_train_transforms_output_shape_and_dtype():
    """Verify training transforms produce expected (224, 224, 3) float32 output."""
    pipeline = AugmentationPipeline(target_size=(224, 224), normalize_to_unit=True)
    train_transform = pipeline.get_train_transforms(enable_bbox=False)

    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    res = train_transform(image=img)

    aug_img = res["image"]
    assert aug_img.shape == (224, 224, 3)
    assert aug_img.dtype == np.float32
    assert 0.0 <= aug_img.min() and aug_img.max() <= 1.0


def test_deterministic_val_and_test_transforms():
    """Verify validation and test transforms are 100% deterministic."""
    pipeline = AugmentationPipeline(target_size=(224, 224), normalize_to_unit=True)
    val_transform = pipeline.get_val_transforms(enable_bbox=False)
    test_transform = pipeline.get_test_transforms(enable_bbox=False)

    img = np.random.randint(0, 255, (180, 250, 3), dtype=np.uint8)

    val_res_1 = val_transform(image=img)["image"]
    val_res_2 = val_transform(image=img)["image"]
    test_res = test_transform(image=img)["image"]

    assert np.array_equal(val_res_1, val_res_2), "Validation transform is non-deterministic!"
    assert np.array_equal(val_res_1, test_res), "Test transform differs from validation transform!"


def test_yolo_bbox_horizontal_flip_transformation():
    """Verify horizontal flip correctly transforms YOLO bounding box x_center (x_new = 1.0 - x_old)."""
    # Create transform pipeline containing ONLY HorizontalFlip and Resize
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True)
    flip_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=1.0),  # Always flip
        ],
        bbox_params=bbox_params,
    )

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # YOLO box: x_center=0.3, y_center=0.4, w=0.2, h=0.3
    bboxes = [(0.3, 0.4, 0.2, 0.3)]
    labels = [0]  # Cattle

    res = flip_transform(image=img, bboxes=bboxes, class_labels=labels)
    flipped_boxes = res["bboxes"]

    assert len(flipped_boxes) == 1
    new_xc, new_yc, new_w, new_h = flipped_boxes[0]

    # After horizontal flip, x_center becomes 1.0 - 0.3 = 0.7
    assert pytest.approx(new_xc, abs=1e-3) == 0.7
    assert pytest.approx(new_yc, abs=1e-3) == 0.4
    assert pytest.approx(new_w, abs=1e-3) == 0.2
    assert pytest.approx(new_h, abs=1e-3) == 0.3


def test_multiple_bboxes_transformation():
    """Verify pipeline transforms multiple YOLO bounding boxes simultaneously."""
    pipeline = AugmentationPipeline(target_size=(224, 224))
    train_transform = pipeline.get_train_transforms(enable_bbox=True)

    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    bboxes = [
        (0.2, 0.3, 0.15, 0.2),  # Box 1
        (0.7, 0.8, 0.2, 0.25),   # Box 2
    ]
    labels = [0, 1]  # Cattle, Buffalo

    res = train_transform(image=img, bboxes=bboxes, class_labels=labels)
    assert "bboxes" in res
    assert "class_labels" in res


def test_empty_bboxes_handling():
    """Verify pipeline handles empty bounding box lists without error."""
    pipeline = AugmentationPipeline(target_size=(224, 224))
    train_transform = pipeline.get_train_transforms(enable_bbox=True)

    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    res = train_transform(image=img, bboxes=[], class_labels=[])

    assert len(res["bboxes"]) == 0
    assert len(res["class_labels"]) == 0
