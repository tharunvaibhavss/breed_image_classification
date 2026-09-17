"""Unit tests for configuration management."""

from app.core.config import settings, Settings


def test_settings_default_values():
    """Test default values of application settings."""
    assert settings.PROJECT_NAME is not None
    assert settings.APP_VERSION == "0.1.0"
    assert settings.DETECTION_MODEL == "YOLO"
    assert settings.CLASSIFICATION_MODEL == "EfficientNet-B0"
    assert settings.EXPLAINABILITY_MODEL == "Grad-CAM"
    assert isinstance(settings.PORT, int)


def test_custom_settings_instantiation():
    """Test custom instantiation of Settings class."""
    custom_settings = Settings(PROJECT_NAME="Test Project", PORT=9000)
    assert custom_settings.PROJECT_NAME == "Test Project"
    assert custom_settings.PORT == 9000


def test_cors_origins_configuration():
    """Test allowed_cors_origins returns configured domain origin list."""
    dev_settings = Settings(APP_ENV="development", CORS_ORIGINS=["http://localhost:3000"])
    assert "http://localhost:3000" in dev_settings.allowed_cors_origins


def test_production_wildcard_cors_rejection():
    """Test ValueError is raised if wildcard '*' origin is used in production mode."""
    import pytest
    prod_settings = Settings(APP_ENV="production", CORS_ORIGINS=["*"])
    with pytest.raises(ValueError, match="Wildcard CORS origin"):
        _ = prod_settings.allowed_cors_origins
