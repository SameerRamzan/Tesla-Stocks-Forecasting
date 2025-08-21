"""Pydantic schemas for Tesla forecasting API."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Base schemas
class ResponseBase(BaseModel):
    """Base response schema."""
    status: str = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")


# Health check
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "0.1.0"


# Data ingestion schemas
class IngestionRequest(BaseModel):
    """Data ingestion request schema."""
    interval: str = Field("1d", description="Data interval (1m, 5m, 1h, 1d, etc.)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    lookback_days: int = Field(30, description="Days to look back if no start_date")

    @validator("interval")
    def validate_interval(cls, v):
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in valid_intervals:
            raise ValueError(f"Interval must be one of {valid_intervals}")
        return v


class IngestionResponse(ResponseBase):
    """Data ingestion response schema."""
    symbol: Optional[str] = None
    interval: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0


# Feature engineering schemas
class FeatureRequest(BaseModel):
    """Feature computation request schema."""
    interval: str = Field("1d", description="Data interval")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    lookback_days: int = Field(100, description="Days of history for feature computation")


class FeatureResponse(ResponseBase):
    """Feature computation response schema."""
    symbol: Optional[str] = None
    interval: Optional[str] = None
    features_computed: int = 0
    features_inserted: int = 0
    features_updated: int = 0


# Price data schemas
class PriceDataPoint(BaseModel):
    """Single price data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    dividends: float = 0.0
    stock_splits: float = 1.0


class PriceDataResponse(BaseModel):
    """Price data response schema."""
    symbol: str
    interval: str
    data: List[PriceDataPoint]
    count: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Feature data schemas
class FeatureDataPoint(BaseModel):
    """Single feature data point."""
    timestamp: datetime
    # Returns
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    log_return_1d: Optional[float] = None
    
    # Rolling statistics
    rolling_mean_5: Optional[float] = None
    rolling_mean_20: Optional[float] = None
    rolling_std_5: Optional[float] = None
    rolling_std_20: Optional[float] = None
    
    # Technical indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    atr_14: Optional[float] = None
    obv: Optional[float] = None
    
    # Calendar features
    day_of_week: int
    month: int
    is_month_end: bool = False
    is_quarter_end: bool = False
    is_year_end: bool = False
    is_holiday: bool = False


class FeatureDataResponse(BaseModel):
    """Feature data response schema."""
    symbol: str
    interval: str
    data: List[FeatureDataPoint]
    count: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Model schemas
class ModelInfo(BaseModel):
    """Model information schema."""
    model_id: UUID
    name: str
    framework: str
    version: str
    stage: str = "Staging"
    active: bool = False
    created_at: datetime
    registered_at: Optional[datetime] = None
    
    # Performance metrics
    cv_mae: Optional[float] = None
    cv_rmse: Optional[float] = None
    cv_mape: Optional[float] = None
    cv_directional_accuracy: Optional[float] = None
    
    # Training metadata
    training_start: Optional[datetime] = None
    training_end: Optional[datetime] = None
    features_used: Optional[List[str]] = None
    target_variable: Optional[str] = None


class ModelListResponse(BaseModel):
    """Model list response schema."""
    models: List[ModelInfo]
    count: int


# Forecast schemas
class ForecastRequest(BaseModel):
    """Forecast request schema."""
    symbol: str = Field("TSLA", description="Stock symbol")
    interval: str = Field("1d", description="Data interval")
    horizons: List[int] = Field([1, 5, 10, 20], description="Forecast horizons")
    return_type: str = Field("close", description="Type of forecast (close, return)")
    include_pi: bool = Field(True, description="Include prediction intervals")
    model_name: Optional[str] = Field(None, description="Specific model name (uses Production if not specified)")

    @validator("horizons")
    def validate_horizons(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one horizon must be specified")
        if any(h <= 0 for h in v):
            raise ValueError("All horizons must be positive")
        return v

    @validator("return_type")
    def validate_return_type(cls, v):
        if v not in ["close", "return"]:
            raise ValueError("return_type must be 'close' or 'return'")
        return v


class ForecastPoint(BaseModel):
    """Single forecast point."""
    target_ts: datetime
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None
    horizon: int


class ForecastResponse(BaseModel):
    """Forecast response schema."""
    model_id: UUID
    model_name: str
    generated_at: datetime
    forecasts: List[ForecastPoint]
    metrics: Optional[Dict[str, float]] = None


# What-if analysis schemas
class FeatureOverride(BaseModel):
    """Feature override for what-if analysis."""
    feature_name: str
    value: Union[float, str]  # Can be absolute value or percentage change like "+10%"
    
    @validator("value")
    def validate_value(cls, v):
        if isinstance(v, str):
            if not (v.endswith("%") and v[:-1].replace("+", "").replace("-", "").replace(".", "").isdigit()):
                raise ValueError("String values must be percentage changes like '+10%' or '-5.5%'")
        return v


class WhatIfRequest(BaseModel):
    """What-if analysis request schema."""
    overrides: List[FeatureOverride] = Field(..., description="Feature overrides")
    horizon: int = Field(5, description="Forecast horizon")
    symbol: str = Field("TSLA", description="Stock symbol")
    interval: str = Field("1d", description="Data interval")
    model_name: Optional[str] = Field(None, description="Specific model name")


class WhatIfResponse(BaseModel):
    """What-if analysis response schema."""
    original_forecast: ForecastPoint
    modified_forecast: ForecastPoint
    overrides_applied: List[FeatureOverride]
    impact: Dict[str, float]  # Difference in predictions


# Backtesting schemas
class BacktestRequest(BaseModel):
    """Backtest request schema."""
    model_names: List[str] = Field(..., description="List of model names to backtest")
    interval: str = Field("1d", description="Data interval")
    initial_train_size: int = Field(100, description="Initial training window size")
    test_size: int = Field(1, description="Test size per iteration")
    max_iterations: int = Field(50, description="Maximum backtest iterations")
    validation_type: str = Field("walk_forward", description="Validation type")
    
    @validator("validation_type")
    def validate_validation_type(cls, v):
        if v not in ["walk_forward", "expanding_window"]:
            raise ValueError("validation_type must be 'walk_forward' or 'expanding_window'")
        return v


class BacktestMetrics(BaseModel):
    """Backtest metrics schema."""
    mae: float
    rmse: float
    mape: float
    smape: float
    directional_accuracy: float
    r2: float
    median_absolute_error: float
    max_error: float
    n_predictions: int


class BacktestResult(BaseModel):
    """Single model backtest result."""
    model_name: str
    status: str
    iterations: int
    total_predictions: int
    metrics: BacktestMetrics
    message: Optional[str] = None


class BacktestResponse(BaseModel):
    """Backtest response schema."""
    results: List[BacktestResult]
    champion_model: Optional[str] = None
    generated_at: datetime


# Data quality schemas
class DataQualityCheck(BaseModel):
    """Data quality check result."""
    check_name: str
    status: str  # pass, warning, fail
    value: Any
    threshold: Optional[Any] = None
    message: Optional[str] = None


class DataQualityReport(BaseModel):
    """Data quality report schema."""
    symbol: str
    interval: str
    record_count: int
    date_range: Dict[str, str]
    checks: List[DataQualityCheck]
    overall_status: str  # pass, warning, fail
    generated_at: datetime


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(100, ge=1, le=1000, description="Page size")


class PaginatedResponse(BaseModel):
    """Paginated response base."""
    page: int
    size: int
    total: int
    pages: int


# Error schemas
class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)