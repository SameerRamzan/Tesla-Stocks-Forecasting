"""Core application configuration."""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    PROJECT_NAME: str = "Tesla Forecasting API"
    PROJECT_VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://tesla_user:tesla_password@localhost:5432/tesla_forecasting"
    DATABASE_URL_SYNC: str = "postgresql://tesla_user:tesla_password@localhost:5432/tesla_forecasting"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://localhost:3000",
        "http://localhost:8080"
    ]
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Data Sources
    YFINANCE_SYMBOL: str = "TSLA"
    DEFAULT_INTERVAL: str = "1d"
    
    # Model Configuration
    MODEL_REGISTRY_STAGE: str = "Production"
    FORECAST_HORIZONS: List[int] = [1, 5, 10, 20]
    
    @validator("FORECAST_HORIZONS", pre=True)
    def parse_forecast_horizons(cls, v: str | List[int]) -> List[int]:
        """Parse forecast horizons from string or list."""
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                # Parse list format: "[1,5,10,20]"
                v = v[1:-1]
            return [int(i.strip()) for i in v.split(",")]
        return v
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Feature Flags
    ENABLE_DEEP_LEARNING: bool = False
    ENABLE_WHAT_IF_SCENARIOS: bool = True
    ENABLE_EXPERIMENT_TRACKING: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False
    
    # Storage (Cloud)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET: Optional[str] = None
    S3_REGION: str = "us-east-1"
    
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GCS_BUCKET: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings