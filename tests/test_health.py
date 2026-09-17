"""Integration tests for GET /api/health endpoint."""

from fastapi.testclient import TestClient


def test_get_health_endpoint(client: TestClient):
    """Test GET /api/health returns 200 OK and expected JSON fields."""
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data
    assert "app_version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert "pytorch_env" in data

    # Verify PyTorch Environment section payload
    pytorch_env = data["pytorch_env"]
    assert "python_version" in pytorch_env
    assert "pytorch_version" in pytorch_env
    assert "cuda_available" in pytorch_env
    assert "compute_device" in pytorch_env
    assert "cpu_fallback" in pytorch_env
