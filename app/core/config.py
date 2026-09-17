"""Application configuration management module using Pydantic Settings."""

import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory path
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Core Application Settings loaded from environment variables or .env file."""

    # Project Information
    PROJECT_NAME: str = Field(
        default="AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes",
        validation_alias="PROJECT_NAME",
    )
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = Field(default="development", validation_alias="APP_ENV")
    DEBUG: bool = Field(default=True, validation_alias="DEBUG")

    # API Server Settings
    HOST: str = Field(default="0.0.0.0", validation_alias="HOST")
    PORT: int = Field(default=8000, validation_alias="PORT")
    LOG_LEVEL: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Deep Learning Pipeline Model Declarations
    DETECTION_MODEL: str = "YOLO"
    CLASSIFICATION_MODEL: str = "EfficientNet-B0"
    EXPLAINABILITY_MODEL: str = "Grad-CAM"
    INFERENCE_BACKEND: str = Field(default="pytorch", validation_alias="INFERENCE_BACKEND")

    # PyTorch / CUDA Settings
    CUDA_VISIBLE_DEVICES: Optional[str] = Field(
        default="0", validation_alias="CUDA_VISIBLE_DEVICES"
    )

    # JWT Security Settings
    SECRET_KEY: str = Field(
        default="09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
        validation_alias="SECRET_KEY",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database Configuration (Foundation for future phases)
    DATABASE_URL: Optional[str] = Field(
        default="postgresql://postgres:postgres@localhost:5432/breed_rec_db",
        validation_alias="DATABASE_URL",
    )

    # MLflow Tracking
    MLFLOW_TRACKING_URI: Optional[str] = Field(
        default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="breed-recognition-experiment",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
        validation_alias="CORS_ORIGINS",
    )

    @property
    def allowed_cors_origins(self) -> list[str]:
        """Return production-safe CORS origin list."""
        origins = self.CORS_ORIGINS
        if isinstance(origins, str):
            import json
            try:
                origins = json.loads(origins)
            except Exception:
                origins = [o.strip() for o in origins.split(",") if o.strip()]
        else:
            origins = list(origins)

        if self.APP_ENV.lower() == "production" and "*" in origins:
            raise ValueError(
                "Wildcard CORS origin '*' is strictly prohibited in production mode. "
                "Please configure explicit domain origins in CORS_ORIGINS."
            )
        return origins

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# Singleton settings instance
settings = Settings()
