"""Database models for Tesla forecasting system."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class PriceData(Base):
    """Raw price data table with TimescaleDB support."""
    
    __tablename__ = "prices"
    
    # Composite primary key: symbol + interval + timestamp
    symbol = Column(String(10), primary_key=True, index=True)
    interval = Column(String(10), primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), primary_key=True, index=True)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Corporate actions
    dividends = Column(Float, default=0.0)
    stock_splits = Column(Float, default=1.0)
    
    # Metadata
    source_ingested_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'interval', 'ts', name='_symbol_interval_ts_uc'),
    )


class FeatureData(Base):
    """Engineered features table."""
    
    __tablename__ = "features"
    
    # Composite primary key: symbol + interval + timestamp
    symbol = Column(String(10), primary_key=True, index=True)
    interval = Column(String(10), primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), primary_key=True, index=True)
    
    # Returns
    return_1d = Column(Float)
    return_5d = Column(Float)
    return_10d = Column(Float)
    return_20d = Column(Float)
    log_return_1d = Column(Float)
    
    # Rolling statistics
    rolling_mean_5 = Column(Float)
    rolling_mean_20 = Column(Float)
    rolling_std_5 = Column(Float)
    rolling_std_20 = Column(Float)
    
    # Technical indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_middle = Column(Float)
    atr_14 = Column(Float)
    obv = Column(Float)
    
    # Calendar features
    day_of_week = Column(Integer)
    month = Column(Integer)
    is_month_end = Column(Boolean, default=False)
    is_quarter_end = Column(Boolean, default=False)
    is_year_end = Column(Boolean, default=False)
    is_holiday = Column(Boolean, default=False)
    
    # Target variables (future values, shifted)
    target_close_1d = Column(Float)
    target_close_5d = Column(Float)
    target_return_1d = Column(Float)
    target_return_5d = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'interval', 'ts', name='_features_symbol_interval_ts_uc'),
    )


class ModelRegistry(Base):
    """Model registry table."""
    
    __tablename__ = "models"
    
    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    framework = Column(String(50), nullable=False)  # arima, xgboost, lstm, etc.
    version = Column(String(50), nullable=False)
    
    # Model parameters and configuration
    params = Column(JSON)
    hyperparams = Column(JSON)
    
    # Training metadata
    training_start = Column(DateTime(timezone=True))
    training_end = Column(DateTime(timezone=True))
    features_used = Column(JSON)  # List of feature names
    target_variable = Column(String(50))
    
    # Performance metrics
    cv_mae = Column(Float)
    cv_rmse = Column(Float)
    cv_mape = Column(Float)
    cv_directional_accuracy = Column(Float)
    
    # Registry metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    registered_at = Column(DateTime(timezone=True))
    active = Column(Boolean, default=False)
    stage = Column(String(20), default="Staging")  # Staging, Production, Archived
    
    # MLflow integration
    mlflow_run_id = Column(String(100))
    mlflow_experiment_id = Column(String(100))
    artifact_uri = Column(Text)
    
    __table_args__ = (
        UniqueConstraint('name', 'version', name='_model_name_version_uc'),
    )


class ForecastData(Base):
    """Forecast results table."""
    
    __tablename__ = "forecasts"
    
    # Composite primary key
    model_id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    symbol = Column(String(10), primary_key=True, index=True)
    interval = Column(String(10), primary_key=True, index=True)
    target_ts = Column(DateTime(timezone=True), primary_key=True, index=True)
    
    # Forecast metadata
    generated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    horizon_steps = Column(Integer, nullable=False)
    horizon_unit = Column(String(10), default="days")
    
    # Predictions
    yhat = Column(Float, nullable=False)
    yhat_lower = Column(Float)  # Lower prediction interval
    yhat_upper = Column(Float)  # Upper prediction interval
    
    # Additional metrics
    prediction_interval = Column(Float, default=0.95)  # 95% confidence interval
    volatility_forecast = Column(Float)
    
    # Metadata
    feature_values = Column(JSON)  # Snapshot of input features
    model_version = Column(String(50))
    
    __table_args__ = (
        UniqueConstraint('model_id', 'symbol', 'interval', 'target_ts', 
                        name='_forecast_model_symbol_interval_target_uc'),
    )


class EventLog(Base):
    """System events and monitoring log."""
    
    __tablename__ = "events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    ts = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Event classification
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, CRITICAL
    component = Column(String(50), nullable=False, index=True)  # ingestion, modeling, api, etc.
    event_type = Column(String(50), nullable=False, index=True)
    
    # Event details
    message = Column(Text, nullable=False)
    payload = Column(JSON)  # Additional structured data
    
    # Context
    user_id = Column(String(100))
    session_id = Column(String(100))
    trace_id = Column(String(100))
    
    # Metrics (for performance monitoring)
    duration_ms = Column(Float)
    memory_mb = Column(Float)
    cpu_percent = Column(Float)