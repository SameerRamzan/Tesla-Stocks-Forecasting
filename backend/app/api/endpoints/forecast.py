"""Forecasting endpoints."""

import logging
from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.app.db.session import get_async_session
from backend.app.models.database import FeatureData, ModelRegistry, ForecastData
from backend.app.models.forecasting import (
    NaiveModel, 
    SimpleMovingAverageModel, 
    ExponentialSmoothingModel,
    ARIMAModel,
    SARIMAModel,
    XGBoostModel
)
from backend.app.schemas.api import (
    ForecastRequest, 
    ForecastResponse, 
    ForecastPoint,
    WhatIfRequest,
    WhatIfResponse,
    BacktestRequest,
    BacktestResponse
)
from backend.app.services.backtesting import BacktestingService
from backend.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# Model registry for quick access
MODEL_REGISTRY = {
    "naive": NaiveModel,
    "sma_5": lambda: SimpleMovingAverageModel(window=5),
    "sma_20": lambda: SimpleMovingAverageModel(window=20),
    "ewma_0.3": lambda: ExponentialSmoothingModel(alpha=0.3),
    "arima_111": lambda: ARIMAModel(order=(1, 1, 1)),
    "sarima_212_111_12": lambda: SARIMAModel(order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)),
    "xgboost": lambda: XGBoostModel()
}


async def get_feature_data(session: AsyncSession, interval: str, limit: int = 100):
    """Get recent feature data for forecasting."""
    query = select(FeatureData).where(
        and_(
            FeatureData.symbol == settings.YFINANCE_SYMBOL,
            FeatureData.interval == interval
        )
    ).order_by(FeatureData.ts.desc()).limit(limit)
    
    result = await session.execute(query)
    return result.scalars().all()


@router.post("/", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Generate forecasts using the specified model or production model.
    """
    try:
        # Determine which model to use
        if request.model_name:
            if request.model_name not in MODEL_REGISTRY:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{request.model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
                )
            model = MODEL_REGISTRY[request.model_name]()
        else:
            # Use default SARIMA model for now
            model = SARIMAModel()
        
        # Get feature data
        feature_records = await get_feature_data(session, request.interval)
        
        if not feature_records:
            raise HTTPException(
                status_code=404,
                detail="No feature data available for forecasting"
            )
        
        # Prepare data for modeling
        import pandas as pd
        
        # For time series models (ARIMA, SARIMA), we only need the target
        if model.framework in ["statsmodels"]:
            target_data = [r.target_close_1d for r in feature_records if r.target_close_1d is not None]
            if len(target_data) < 20:
                raise HTTPException(
                    status_code=400,
                    detail="Insufficient data for time series modeling"
                )
            
            target_series = pd.Series(target_data[::-1])  # Reverse to chronological order
            feature_df = pd.DataFrame()  # Empty for time series models
        else:
            # For ML models, prepare features
            feature_cols = [
                'return_1d', 'return_5d', 'rolling_mean_5', 'rolling_mean_20',
                'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14', 'day_of_week', 'month'
            ]
            
            data_for_ml = []
            targets = []
            
            for record in feature_records:
                if record.target_close_1d is not None:
                    row = {}
                    for col in feature_cols:
                        value = getattr(record, col, None)
                        row[col] = value if value is not None else 0.0
                    data_for_ml.append(row)
                    targets.append(record.target_close_1d)
            
            if len(data_for_ml) < 20:
                raise HTTPException(
                    status_code=400,
                    detail="Insufficient data for ML modeling"
                )
            
            feature_df = pd.DataFrame(data_for_ml)
            target_series = pd.Series(targets)
        
        # Fit model
        fit_result = model.fit(feature_df, target_series)
        if fit_result.get("status") != "success":
            raise HTTPException(
                status_code=400,
                detail=f"Model fitting failed: {fit_result.get('message', 'Unknown error')}"
            )
        
        # Generate predictions for each horizon
        forecasts = []
        base_time = datetime.utcnow()
        
        for horizon in request.horizons:
            try:
                if model.framework == "statsmodels":
                    # For time series models, predict the horizon directly
                    pred_result = model.predict(feature_df, horizon=horizon)
                else:
                    # For ML models, use the latest features
                    latest_features = feature_df.tail(1) if not feature_df.empty else pd.DataFrame()
                    pred_result = model.predict(latest_features, horizon=1)
                
                predictions = pred_result["predictions"]
                prediction_intervals = pred_result.get("prediction_intervals")
                
                # Create forecast point
                forecast_point = ForecastPoint(
                    target_ts=base_time,  # Simplified for now
                    yhat=predictions[0] if predictions else 0.0,
                    yhat_lower=prediction_intervals["lower"][0] if prediction_intervals else None,
                    yhat_upper=prediction_intervals["upper"][0] if prediction_intervals else None,
                    horizon=horizon
                )
                forecasts.append(forecast_point)
                
            except Exception as e:
                logger.warning(f"Error generating forecast for horizon {horizon}: {str(e)}")
                continue
        
        if not forecasts:
            raise HTTPException(
                status_code=400,
                detail="No forecasts could be generated"
            )
        
        return ForecastResponse(
            model_id=uuid4(),  # Generate a temporary ID
            model_name=model.name,
            generated_at=datetime.utcnow(),
            forecasts=forecasts,
            metrics=None  # Could add cross-validation metrics here
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whatif", response_model=WhatIfResponse)
async def what_if_analysis(
    request: WhatIfRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Perform what-if analysis by modifying features and re-forecasting.
    """
    try:
        # This is a simplified implementation
        # In a full system, you would:
        # 1. Get the latest feature data
        # 2. Apply the overrides
        # 3. Generate forecast with modified features
        # 4. Compare to baseline forecast
        
        raise HTTPException(
            status_code=501,
            detail="What-if analysis not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in what-if analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Run backtesting for specified models.
    """
    try:
        # Get feature data for backtesting
        feature_records = await get_feature_data(session, request.interval, limit=500)
        
        if not feature_records:
            raise HTTPException(
                status_code=404,
                detail="No feature data available for backtesting"
            )
        
        # Prepare data
        import pandas as pd
        
        # Convert to DataFrame
        data_rows = []
        for record in reversed(feature_records):  # Chronological order
            if record.target_close_1d is not None:
                row = {
                    'ts': record.ts,
                    'target_close_1d': record.target_close_1d,
                    'return_1d': record.return_1d or 0.0,
                    'return_5d': record.return_5d or 0.0,
                    'rolling_mean_5': record.rolling_mean_5 or 0.0,
                    'rolling_mean_20': record.rolling_mean_20 or 0.0,
                    'rsi_14': record.rsi_14 or 50.0,
                    'macd': record.macd or 0.0,
                    'bb_upper': record.bb_upper or 0.0,
                    'bb_lower': record.bb_lower or 0.0,
                    'atr_14': record.atr_14 or 0.0,
                    'day_of_week': record.day_of_week or 0,
                    'month': record.month or 1
                }
                data_rows.append(row)
        
        if len(data_rows) < request.initial_train_size + 10:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for backtesting"
            )
        
        df = pd.DataFrame(data_rows)
        df.set_index('ts', inplace=True)
        
        # Run backtesting
        backtesting_service = BacktestingService()
        results = []
        
        feature_columns = [
            'return_1d', 'return_5d', 'rolling_mean_5', 'rolling_mean_20',
            'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14', 'day_of_week', 'month'
        ]
        
        for model_name in request.model_names:
            if model_name not in MODEL_REGISTRY:
                logger.warning(f"Model '{model_name}' not found, skipping")
                continue
            
            try:
                model = MODEL_REGISTRY[model_name]()
                
                if request.validation_type == "walk_forward":
                    result = backtesting_service.walk_forward_validation(
                        model=model,
                        data=df,
                        target_column='target_close_1d',
                        feature_columns=feature_columns,
                        initial_train_size=request.initial_train_size,
                        test_size=request.test_size,
                        max_iterations=request.max_iterations
                    )
                else:
                    result = backtesting_service.expanding_window_validation(
                        model=model,
                        data=df,
                        target_column='target_close_1d',
                        feature_columns=feature_columns,
                        initial_train_size=request.initial_train_size,
                        test_size=request.test_size,
                        max_iterations=request.max_iterations
                    )
                
                if result.get("status") == "success":
                    from backend.app.schemas.api import BacktestResult, BacktestMetrics
                    
                    backtest_result = BacktestResult(
                        model_name=result["model_name"],
                        status=result["status"],
                        iterations=result["iterations"],
                        total_predictions=result["total_predictions"],
                        metrics=BacktestMetrics(**result["metrics"]),
                        message=result.get("message")
                    )
                    results.append(backtest_result)
                
            except Exception as e:
                logger.error(f"Error backtesting model {model_name}: {str(e)}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=400,
                detail="No successful backtest results"
            )
        
        # Find champion model (lowest RMSE)
        champion = min(results, key=lambda x: x.metrics.rmse)
        
        return BacktestResponse(
            results=results,
            champion_model=champion.model_name,
            generated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))