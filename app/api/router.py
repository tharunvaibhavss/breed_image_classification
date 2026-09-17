"""Main API router combining endpoint sub-routers under /api."""

from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.breeds import router as breeds_router
from app.api.model_info import router as model_info_router
from app.api.auth import router as auth_router
from app.api.admin import router as admin_router

api_router = APIRouter()

# Include endpoint sub-routers under /api prefix
api_router.include_router(health_router)
api_router.include_router(predict_router)
api_router.include_router(breeds_router)
api_router.include_router(model_info_router)
api_router.include_router(auth_router)
api_router.include_router(admin_router)
