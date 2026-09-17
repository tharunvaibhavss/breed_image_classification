"""Unit tests for OpenCV image preprocessing pipeline."""

import pytest
import numpy as np
import cv2
from pathlib import Path

from ml.preprocessing.opencv_pipeline import OpenCVPreprocessor


def test_rgb_image_preprocessing():
    """Verify preprocessing of standard 3-channel RGB image."""
    preprocessor = OpenCVPreprocessor(target_size=(224, 224))
    
    # 300x400 BGR numpy array
    rgb_arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    np_out = preprocessor.preprocess_to_numpy(rgb_arr)
    assert isinstance(np_out, np.ndarray)
    assert np_out.shape == (224, 224, 3)
    assert np_out.dtype == np.float32
    assert 0.0 <= np_out.min() and np_out.max() <= 1.0


def test_grayscale_image_preprocessing():
    """Verify grayscale (1-channel or 2D) image is converted to 3-channel RGB."""
    preprocessor = OpenCVPreprocessor(target_size=(224, 224))
    
    # 2D Grayscale array
    gray_arr = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    
    np_out = preprocessor.preprocess_to_numpy(gray_arr)
    assert np_out.shape == (224, 224, 3)
    assert np_out.dtype == np.float32

    # 3D 1-channel array (150, 150, 1)
    gray_3d = gray_arr[:, :, np.newaxis]
    np_out_3d = preprocessor.preprocess_to_numpy(gray_3d)
    assert np_out_3d.shape == (224, 224, 3)


def test_invalid_image_inputs(tmp_path: Path):
    """Verify exception handling for empty, non-existent, or corrupted image inputs."""
    preprocessor = OpenCVPreprocessor()

    # 1. Non-existent file path
    with pytest.raises(ValueError, match="path does not exist"):
        preprocessor.preprocess_to_numpy(tmp_path / "non_existent.jpg")

    # 2. Corrupted in-memory bytes
    with pytest.raises(ValueError, match="Failed to decode"):
        preprocessor.preprocess_to_numpy(b"INVALID_CORRUPTED_BYTES")

    # 3. Empty numpy array
    with pytest.raises(ValueError, match="empty or has fewer than 2 dimensions"):
        preprocessor.preprocess_to_numpy(np.array([]))


def test_different_image_resolutions():
    """Verify pipeline handles various resolutions (tiny, large, wide, tall)."""
    preprocessor = OpenCVPreprocessor(target_size=(224, 224))

    resolutions = [
        (10, 10),        # Tiny square
        (2000, 2000),    # Ultra large square
        (1920, 1080),    # Wide aspect ratio (16:9)
        (1080, 1920),    # Tall portrait aspect ratio (9:16)
        (64, 400),       # Extreme rectangle
    ]

    for h, w in resolutions:
        img_arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        np_out = preprocessor.preprocess_to_numpy(img_arr)
        
        assert np_out.shape == (224, 224, 3)
        assert np_out.dtype == np.float32


def test_output_shape_and_dtype():
    """Verify exact output shapes and data types for numpy array and PyTorch tensor."""
    preprocessor = OpenCVPreprocessor(target_size=(224, 224))
    img_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Numpy output test
    np_out = preprocessor.preprocess_to_numpy(img_arr)
    assert np_out.shape == (224, 224, 3)
    assert np_out.dtype == np.float32

    # PyTorch Tensor output test
    try:
        import torch
        tensor_out = preprocessor.preprocess_to_tensor(img_arr)
        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.shape == (3, 224, 224)
        assert tensor_out.dtype == torch.float32
    except ImportError:
        pass


def test_training_and_inference_pipeline_compatibility():
    """Verify that training and inference preprocessing produce identical outputs for identical inputs."""
    train_preprocessor = OpenCVPreprocessor(target_size=(224, 224), keep_aspect_ratio=True)
    infer_preprocessor = OpenCVPreprocessor(target_size=(224, 224), keep_aspect_ratio=True)

    img_arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    train_out = train_preprocessor.preprocess_to_numpy(img_arr)
    infer_out = infer_preprocessor.preprocess_to_numpy(img_arr)

    assert np.array_equal(train_out, infer_out), "Training and inference preprocessing outputs are incompatible!"


def test_roi_cropping():
    """Verify ROI bounding box crop functionality."""
    preprocessor = OpenCVPreprocessor(target_size=(224, 224))
    img_arr = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    # Crop bounding box (xmin=50, ymin=50, xmax=250, ymax=250)
    bbox = (50, 50, 250, 250)
    cropped_out = preprocessor.preprocess_to_numpy(img_arr, bbox=bbox)

    assert cropped_out.shape == (224, 224, 3)
    assert cropped_out.dtype == np.float32
