"""Unit tests for EfficientNet-B0 breed classification model, dataset loader, metrics, and predictor."""

import numpy as np
import pytest
import torch
from pathlib import Path

from ml.classification.efficientnet import BreedClassifier
from ml.classification.dataset import BreedDataset
from ml.classification.metrics import calculate_classification_metrics, calculate_topk_accuracy
from ml.classification.predictor import BreedPredictor, PredictionResult
from ml.classification.trainer import set_reproducible_seeds


def test_efficientnet_model_forward_pass():
    """Verify BreedClassifier forward pass outputs shape (batch_size, 6)."""
    model = BreedClassifier(num_classes=6, pretrained=False)
    model.eval()

    # Batch of 2 images, 3 channels, 224x224
    dummy_batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_batch)

    assert logits.shape == (2, 6)
    assert logits.dtype == torch.float32


def test_breed_dataset_loader(tmp_path: Path):
    """Verify BreedDataset returns float32 tensor (3, 224, 224) and integer label."""
    # Create dummy sample record
    samples = [
        {"file_path": "cattle/Gir/gir_01.jpg", "breed": "Gir", "class_id": 0},
        {"file_path": "buffalo/Murrah/murrah_01.jpg", "breed": "Murrah", "class_id": 4},
    ]

    dataset = BreedDataset(samples=samples, base_dir=tmp_path, target_size=(224, 224))
    assert len(dataset) == 2

    img_tensor, class_id = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert img_tensor.dtype == torch.float32
    assert class_id == 0

    img_tensor_2, class_id_2 = dataset[1]
    assert class_id_2 == 4


def test_classification_metrics_calculation():
    """Verify accuracy, Top-3, precision, recall, and F1 calculations."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])

    # Perfect prediction logits
    y_pred_logits = np.zeros((10, 6), dtype=np.float32)
    for i, t in enumerate(y_true):
        y_pred_logits[i, t] = 10.0

    metrics = calculate_classification_metrics(y_true, y_pred_logits, num_classes=6)

    assert metrics["accuracy"] == 100.0
    assert metrics["top_3_accuracy"] == 100.0
    assert metrics["f1_macro"] == 100.0
    assert metrics["f1_weighted"] == 100.0


def test_topk_accuracy_calculator():
    """Verify Top-1 and Top-3 accuracy calculation logic."""
    logits = np.array(
        [
            [10.0, 2.0, 1.0, 0.0, 0.0, 0.0],  # Top-1 is class 0
            [1.0, 2.0, 10.0, 0.0, 0.0, 0.0],  # Top-1 is class 2
        ],
        dtype=np.float32,
    )
    targets = np.array([0, 2])

    topk_res = calculate_topk_accuracy(logits, targets, topk=(1, 3))
    assert topk_res["top_1_acc"] == 100.0
    assert topk_res["top_3_acc"] == 100.0


def test_breed_predictor_inference():
    """Verify BreedPredictor returns PredictionResult with Top-1 and Top-3 predictions."""
    predictor = BreedPredictor()
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    res = predictor.predict(dummy_img, top_k=3)

    assert isinstance(res, PredictionResult)
    assert res.predicted_breed in ["Gir", "Ongole", "Sahiwal", "Jaffarabadi", "Murrah", "Surti"]
    assert res.animal_type in ["cattle", "buffalo"]
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_3_predictions) == 3

    # Check that Top-3 probabilities sum to <= 1.0
    top_3_confs = [item.confidence for item in res.top_3_predictions]
    assert sum(top_3_confs) <= 1.0001
    assert top_3_confs[0] >= top_3_confs[1] >= top_3_confs[2]


def test_reproducible_seed_setting():
    """Verify set_reproducible_seeds fixes random states."""
    set_reproducible_seeds(seed=42)
    val_1 = torch.randn(2, 2)

    set_reproducible_seeds(seed=42)
    val_2 = torch.randn(2, 2)

    assert torch.equal(val_1, val_2), "Random seed setting is non-deterministic!"
