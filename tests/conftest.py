"""Pytest shared fixtures."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Provide FastAPI TestClient instance for API tests."""
    with TestClient(app) as test_client:
        yield test_client
