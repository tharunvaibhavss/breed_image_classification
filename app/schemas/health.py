"""Pydantic schema models for API health checks."""

from typing import Dict, Any
from pydantic import BaseModel, Field


class PyTorchEnvStatus(BaseModel):
    """Schema representing PyTorch execution environment details."""

    status: str = Field(description="Environment status flag")
    python_version: str = Field(description="Python runtime version")
    platform: str = Field(description="Host OS and hardware architecture")
    pytorch_version: str = Field(description="PyTorch package version")
    cuda_available: bool = Field(description="CUDA hardware acceleration availability")
    cuda_version: str | None = Field(default=None, description="CUDA driver/runtime version")
    device_count: int = Field(default=0, description="Count of detected GPU devices")
    gpu_name: str | None = Field(default=None, description="Primary GPU device name")
    compute_device: str = Field(description="Active compute device (cuda:0 or cpu)")
    cpu_fallback: bool = Field(description="True if system is operating on CPU fallback")


class HealthResponse(BaseModel):
    """Schema for GET /api/health endpoint response."""

    status: str = Field(default="ok", description="Service operational status")
    app_name: str = Field(description="Application title")
    app_version: str = Field(description="Application version string")
    environment: str = Field(description="Execution environment (development/production)")
    timestamp: str = Field(description="ISO-8601 UTC response timestamp")
    pytorch_env: PyTorchEnvStatus = Field(description="PyTorch environment diagnostic payload")
