"""Main FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.router import api_router
from app.middleware.request_id import RequestIDMiddleware
from ml.common.env_check import get_pytorch_environment_info, format_env_summary

# Initialize logger
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Starting %s v%s...", settings.PROJECT_NAME, settings.APP_VERSION)
    logger.info("Current Phase: Phase 14 - Authentication & Administration")

    # Initialize database tables
    try:
        from db.session import init_db
        init_db()
    except Exception as e:
        logger.warning("Database initialization notice: %s", e)

    # Detect PyTorch Environment on startup
    env_info = get_pytorch_environment_info()
    logger.info("PyTorch Environment Check on Startup:\n%s", format_env_summary(env_info))

    yield

    logger.info("Shutting down %s...", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """Application factory for constructing FastAPI instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes "
            "using Deep Learning (YOLO + EfficientNet-B0 + Grad-CAM)"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure Request ID Middleware
    app.add_middleware(RequestIDMiddleware)

    # Register API Router with /api prefix
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
