"""Health check endpoints."""

from fastapi import APIRouter
from backend.app.schemas.api import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()