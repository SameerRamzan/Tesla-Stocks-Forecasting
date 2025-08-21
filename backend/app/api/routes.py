"""API routes for Tesla forecasting system."""

from fastapi import APIRouter

from backend.app.api.endpoints import (
    health,
    ingestion,
    features,
    models,
    forecast,
    data
)

router = APIRouter()

# Include all endpoint routers
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(ingestion.router, prefix="/ingestion", tags=["data-ingestion"])
router.include_router(features.router, prefix="/features", tags=["feature-engineering"])
router.include_router(models.router, prefix="/models", tags=["models"])
router.include_router(forecast.router, prefix="/forecast", tags=["forecasting"])
router.include_router(data.router, prefix="/data", tags=["data"])