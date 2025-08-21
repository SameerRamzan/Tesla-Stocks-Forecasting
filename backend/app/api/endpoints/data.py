"""Data access endpoints."""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.app.db.session import get_async_session
from backend.app.models.database import PriceData, FeatureData
from backend.app.schemas.api import (
    PriceDataResponse, 
    PriceDataPoint,
    FeatureDataResponse,
    FeatureDataPoint,
    PaginationParams
)
from backend.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/prices", response_model=PriceDataResponse)
async def get_prices(
    interval: str = Query("1d", description="Data interval"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, le=5000, description="Maximum number of records"),
    session: AsyncSession = Depends(get_async_session)
):
    """Get historical price data."""
    try:
        # Build query
        query = select(PriceData).where(
            and_(
                PriceData.symbol == settings.YFINANCE_SYMBOL,
                PriceData.interval == interval
            )
        )
        
        if start_date:
            query = query.where(PriceData.ts >= start_date)
        if end_date:
            query = query.where(PriceData.ts <= end_date)
            
        query = query.order_by(PriceData.ts.desc()).limit(limit)
        
        result = await session.execute(query)
        records = result.scalars().all()
        
        # Convert to response format
        data_points = [
            PriceDataPoint(
                timestamp=record.ts,
                open=record.open,
                high=record.high,
                low=record.low,
                close=record.close,
                adj_close=record.adj_close,
                volume=record.volume,
                dividends=record.dividends,
                stock_splits=record.stock_splits
            )
            for record in records
        ]
        
        return PriceDataResponse(
            symbol=settings.YFINANCE_SYMBOL,
            interval=interval,
            data=data_points,
            count=len(data_points),
            start_date=start_date,
            end_date=end_date
        )
        
    except Exception as e:
        logger.error(f"Error getting price data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features", response_model=FeatureDataResponse)
async def get_features(
    interval: str = Query("1d", description="Data interval"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, le=5000, description="Maximum number of records"),
    session: AsyncSession = Depends(get_async_session)
):
    """Get computed feature data."""
    try:
        # Build query
        query = select(FeatureData).where(
            and_(
                FeatureData.symbol == settings.YFINANCE_SYMBOL,
                FeatureData.interval == interval
            )
        )
        
        if start_date:
            query = query.where(FeatureData.ts >= start_date)
        if end_date:
            query = query.where(FeatureData.ts <= end_date)
            
        query = query.order_by(FeatureData.ts.desc()).limit(limit)
        
        result = await session.execute(query)
        records = result.scalars().all()
        
        # Convert to response format
        data_points = [
            FeatureDataPoint(
                timestamp=record.ts,
                return_1d=record.return_1d,
                return_5d=record.return_5d,
                return_10d=record.return_10d,
                return_20d=record.return_20d,
                log_return_1d=record.log_return_1d,
                rolling_mean_5=record.rolling_mean_5,
                rolling_mean_20=record.rolling_mean_20,
                rolling_std_5=record.rolling_std_5,
                rolling_std_20=record.rolling_std_20,
                rsi_14=record.rsi_14,
                macd=record.macd,
                macd_signal=record.macd_signal,
                macd_histogram=record.macd_histogram,
                bb_upper=record.bb_upper,
                bb_lower=record.bb_lower,
                bb_middle=record.bb_middle,
                atr_14=record.atr_14,
                obv=record.obv,
                day_of_week=record.day_of_week,
                month=record.month,
                is_month_end=record.is_month_end,
                is_quarter_end=record.is_quarter_end,
                is_year_end=record.is_year_end,
                is_holiday=record.is_holiday
            )
            for record in records
        ]
        
        return FeatureDataResponse(
            symbol=settings.YFINANCE_SYMBOL,
            interval=interval,
            data=data_points,
            count=len(data_points),
            start_date=start_date,
            end_date=end_date
        )
        
    except Exception as e:
        logger.error(f"Error getting feature data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))