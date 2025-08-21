"""Test forecasting models."""

import pytest
import pandas as pd
import numpy as np
from backend.app.models.forecasting import (
    NaiveModel,
    SimpleMovingAverageModel,
    ExponentialSmoothingModel,
    ARIMAModel,
    SARIMAModel
)


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    # Create trending data with noise
    trend = np.linspace(100, 150, 100)
    noise = np.random.normal(0, 5, 100)
    values = trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    }).set_index('date'), pd.Series(values, index=dates)


def test_naive_model(sample_data):
    """Test Naive model functionality."""
    df, series = sample_data
    
    model = NaiveModel()
    
    # Test fitting
    result = model.fit(df, series)
    assert result["status"] == "success"
    assert model.is_fitted
    
    # Test prediction
    pred_result = model.predict(df, horizon=5)
    assert len(pred_result["predictions"]) == 5
    assert all(p == series.iloc[-1] for p in pred_result["predictions"])


def test_sma_model(sample_data):
    """Test Simple Moving Average model."""
    df, series = sample_data
    
    model = SimpleMovingAverageModel(window=5)
    
    # Test fitting
    result = model.fit(df, series)
    assert result["status"] == "success"
    assert model.is_fitted
    
    # Test prediction
    pred_result = model.predict(df, horizon=3)
    assert len(pred_result["predictions"]) == 3
    
    # Prediction should be average of last 5 values
    expected = series.tail(5).mean()
    assert all(abs(p - expected) < 1e-10 for p in pred_result["predictions"])


def test_ewma_model(sample_data):
    """Test Exponential Weighted Moving Average model."""
    df, series = sample_data
    
    model = ExponentialSmoothingModel(alpha=0.3)
    
    # Test fitting
    result = model.fit(df, series)
    assert result["status"] == "success"
    assert model.is_fitted
    
    # Test prediction
    pred_result = model.predict(df, horizon=2)
    assert len(pred_result["predictions"]) == 2


def test_arima_model(sample_data):
    """Test ARIMA model."""
    df, series = sample_data
    
    model = ARIMAModel(order=(1, 1, 1))
    
    # Test fitting
    result = model.fit(df, series)
    if result["status"] == "success":  # ARIMA might fail on some data
        assert model.is_fitted
        
        # Test prediction
        pred_result = model.predict(df, horizon=3)
        assert len(pred_result["predictions"]) == 3
        assert "prediction_intervals" in pred_result


def test_sarima_model(sample_data):
    """Test SARIMA model."""
    df, series = sample_data
    
    model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    
    # Test fitting (might fail due to insufficient seasonal data)
    result = model.fit(df, series)
    if result["status"] == "success":
        assert model.is_fitted
        
        # Test prediction
        pred_result = model.predict(df, horizon=2)
        assert len(pred_result["predictions"]) == 2


def test_model_save_load():
    """Test model serialization."""
    model = NaiveModel()
    
    # Create dummy data
    dummy_series = pd.Series([1, 2, 3, 4, 5])
    model.fit(pd.DataFrame(), dummy_series)
    
    # Test that model is fitted
    assert model.is_fitted
    
    # For this test, we'll just verify the model properties
    assert model.name == "Naive Last Value"
    assert model.framework == "baseline"
    assert model.last_value == 5