"""Feature engineering endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_async_session
from backend.app.services.features import FeatureEngineeringService
from backend.app.schemas.api import FeatureRequest, FeatureResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/compute", response_model=FeatureResponse)
async def compute_features(
    request: FeatureRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Compute features from price data.
    
    This endpoint computes technical indicators, rolling statistics,
    calendar features, and target variables from raw price data.
    """
    try:
        service = FeatureEngineeringService(session)
        
        result = await service.compute_and_store_features(
            interval=request.interval,
            start_date=request.start_date,
            end_date=request.end_date,
            lookback_days=request.lookback_days
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return FeatureResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in feature computation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))