"""Health check endpoint router."""

from datetime import datetime, timezone
from fastapi import APIRouter, status
from app.core.config import settings
from app.schemas.health import HealthResponse, PyTorchEnvStatus
from ml.common.env_check import get_pytorch_environment_info

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns the operational status of the service along with PyTorch execution environment diagnostic details.",
)
async def get_health() -> HealthResponse:
    """Check health and PyTorch environment status."""
    pytorch_info = get_pytorch_environment_info()

    return HealthResponse(
        status="ok",
        app_name=settings.PROJECT_NAME,
        app_version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
        pytorch_env=PyTorchEnvStatus(**pytorch_info),
    )
