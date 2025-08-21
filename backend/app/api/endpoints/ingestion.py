"""Data ingestion endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_async_session
from backend.app.services.ingestion import DataIngestionService
from backend.app.schemas.api import IngestionRequest, IngestionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_data(
    request: IngestionRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Ingest Tesla stock data from yfinance.
    
    This endpoint fetches historical stock data and stores it in the database.
    It's idempotent and will only insert new data or update existing records.
    """
    try:
        service = DataIngestionService(session)
        
        result = await service.fetch_and_store_data(
            interval=request.interval,
            start_date=request.start_date,
            end_date=request.end_date,
            lookback_days=request.lookback_days
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return IngestionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-quality/{interval}")
async def get_data_quality(
    interval: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Get data quality summary for the specified interval."""
    try:
        service = DataIngestionService(session)
        result = await service.get_data_quality_summary(interval)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting data quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest-timestamp/{interval}")
async def get_latest_timestamp(
    interval: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Get the timestamp of the most recent data for the given interval."""
    try:
        service = DataIngestionService(session)
        latest_ts = await service.get_latest_data_timestamp(interval)
        
        return {
            "interval": interval,
            "latest_timestamp": latest_ts.isoformat() if latest_ts else None
        }
        
    except Exception as e:
        logger.error(f"Error getting latest timestamp: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))