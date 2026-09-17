"""Unit and integration tests for FastAPI REST API endpoints."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_get_health_endpoint():
    """Verify GET /api/health endpoint response and X-Request-ID header."""
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "X-Request-ID" in response.headers


def test_get_breeds_endpoint():
    """Verify GET /api/breeds returns catalog of supported Indian cattle and buffalo breeds."""
    response = client.get("/api/breeds")

    assert response.status_code == 200
    data = response.json()
    assert data["total_breeds"] == 6
    assert len(data["breeds"]) == 6

    breed_names = [b["breed_name"] for b in data["breeds"]]
    assert "Gir" in breed_names
    assert "Ongole" in breed_names
    assert "Sahiwal" in breed_names
    assert "Jaffarabadi" in breed_names
    assert "Murrah" in breed_names
    assert "Surti" in breed_names


def test_get_model_info_endpoint():
    """Verify GET /api/model-info returns system metadata and PyTorch env status."""
    response = client.get("/api/model-info")

    assert response.status_code == 200
    data = response.json()
    assert data["pipeline_version"] == "1.0.0"
    assert data["supported_species"] == ["cattle", "buffalo"]
    assert data["supported_breeds_count"] == 6
    assert "checkpoints_status" in data
    assert "environment_info" in data


def test_post_predict_valid_image():
    """Verify POST /api/predict with valid PNG image file upload."""
    # Create valid synthetic RGB test image buffer
    img = np.random.randint(40, 220, (250, 250, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)
    img_bytes = buffer.tobytes()

    files = {"file": ("test_cow.png", img_bytes, "image/png")}
    response = client.post("/api/predict?generate_gradcam=true", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "animal_type" in data
    assert "animal_confidence" in data
    assert "bounding_box" in data
    assert "predicted_breed" in data
    assert "breed_confidence" in data
    assert len(data["top_3_predictions"]) == 3
    assert "prediction_status" in data
    assert "inference_time" in data
    assert "model_versions" in data
    assert "X-Request-ID" in response.headers


def test_post_predict_invalid_file_format():
    """Verify POST /api/predict rejects invalid file types with 400 Bad Request."""
    text_bytes = b"Hello, this is a plain text file, not an image."
    files = {"file": ("document.txt", text_bytes, "text/plain")}

    response = client.post("/api/predict", files=files)

    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


def test_post_predict_oversized_file():
    """Verify POST /api/predict rejects files exceeding 10MB limit with 413 Payload Too Large."""
    # 11 MB dummy buffer
    large_bytes = b"0" * (11 * 1024 * 1024)
    files = {"file": ("large_image.jpg", large_bytes, "image/jpeg")}

    response = client.post("/api/predict", files=files)

    assert response.status_code == 413
    assert "exceeds maximum limit of 10MB" in response.json()["detail"]
