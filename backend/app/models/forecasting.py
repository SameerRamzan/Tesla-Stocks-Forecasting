"""Base model interface and implementations for Tesla forecasting."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, name: str, framework: str):
        self.name = name
        self.framework = framework
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.training_metadata = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Generate predictions."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        pass

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "model": self.model,
            "name": self.name,
            "framework": self.framework,
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata,
            "is_fitted": self.is_fitted
        }
        joblib.dump(model_data, path)

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.name = model_data["name"]
        self.framework = model_data["framework"]
        self.feature_names = model_data["feature_names"]
        self.training_metadata = model_data["training_metadata"]
        self.is_fitted = model_data["is_fitted"]


class NaiveModel(BaseModel):
    """Naive forecasting model - uses last known value."""

    def __init__(self):
        super().__init__("Naive Last Value", "baseline")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit naive model (just store last value)."""
        self.last_value = y.iloc[-1]
        self.is_fitted = True
        self.training_metadata = {
            "training_samples": len(y),
            "last_value": float(self.last_value),
            "fitted_at": datetime.utcnow().isoformat()
        }
        
        return {"status": "success", "message": "Naive model fitted"}

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Predict using last known value."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = [self.last_value] * horizon
        
        return {
            "predictions": predictions,
            "prediction_intervals": None,
            "horizon": horizon,
            "model_name": self.name
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Naive model has no features."""
        return None


class SimpleMovingAverageModel(BaseModel):
    """Simple Moving Average model."""

    def __init__(self, window: int = 5):
        super().__init__(f"SMA_{window}", "baseline")
        self.window = window

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit SMA model."""
        if len(y) < self.window:
            raise ValueError(f"Need at least {self.window} observations for SMA")
        
        self.historical_values = y.tail(self.window).values
        self.is_fitted = True
        self.training_metadata = {
            "window": self.window,
            "training_samples": len(y),
            "fitted_at": datetime.utcnow().isoformat()
        }
        
        return {"status": "success", "message": f"SMA model fitted with window {self.window}"}

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Predict using simple moving average."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # For SMA, prediction is the average of last window values
        prediction = np.mean(self.historical_values)
        predictions = [prediction] * horizon
        
        return {
            "predictions": predictions,
            "prediction_intervals": None,
            "horizon": horizon,
            "model_name": self.name
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """SMA model has no features."""
        return None


class ExponentialSmoothingModel(BaseModel):
    """Exponential Weighted Moving Average model."""

    def __init__(self, alpha: float = 0.3):
        super().__init__(f"EWMA_{alpha}", "baseline")
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit EWMA model."""
        # Compute exponentially weighted moving average
        ewma = y.ewm(alpha=self.alpha).mean()
        self.last_ewma = ewma.iloc[-1]
        self.is_fitted = True
        
        self.training_metadata = {
            "alpha": self.alpha,
            "training_samples": len(y),
            "last_ewma": float(self.last_ewma),
            "fitted_at": datetime.utcnow().isoformat()
        }
        
        return {"status": "success", "message": f"EWMA model fitted with alpha {self.alpha}"}

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Predict using exponential smoothing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # For EWMA, prediction is the last smoothed value
        predictions = [self.last_ewma] * horizon
        
        return {
            "predictions": predictions,
            "prediction_intervals": None,
            "horizon": horizon,
            "model_name": self.name
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """EWMA model has no features."""
        return None


class ARIMAModel(BaseModel):
    """ARIMA model implementation."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__(f"ARIMA_{order[0]}_{order[1]}_{order[2]}", "statsmodels")
        self.order = order

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit ARIMA model."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Fit ARIMA model
            self.model = ARIMA(y, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # Store training metadata
            self.training_metadata = {
                "order": self.order,
                "training_samples": len(y),
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
                "fitted_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"ARIMA model fitted with AIC: {self.fitted_model.aic:.4f}")
            
            return {
                "status": "success", 
                "message": f"ARIMA{self.order} model fitted",
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic
            }
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            return {"status": "error", "message": str(e)}

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Generate ARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate forecast with confidence intervals
            forecast = self.fitted_model.forecast(steps=horizon, alpha=0.05)
            conf_int = self.fitted_model.get_forecast(steps=horizon).conf_int()
            
            return {
                "predictions": forecast.tolist(),
                "prediction_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "horizon": horizon,
                "model_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error generating ARIMA predictions: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """ARIMA model has no external features."""
        return None


class SARIMAModel(BaseModel):
    """SARIMA model implementation."""

    def __init__(self, 
                 order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        super().__init__(f"SARIMA_{order[0]}_{order[1]}_{order[2]}_{seasonal_order[0]}_{seasonal_order[1]}_{seasonal_order[2]}_{seasonal_order[3]}", "statsmodels")
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit SARIMA model."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Fit SARIMA model
            self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            self.is_fitted = True
            
            # Store training metadata
            self.training_metadata = {
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "training_samples": len(y),
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
                "fitted_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"SARIMA model fitted with AIC: {self.fitted_model.aic:.4f}")
            
            return {
                "status": "success", 
                "message": f"SARIMA{self.order}x{self.seasonal_order} model fitted",
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic
            }
            
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            return {"status": "error", "message": str(e)}

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Generate SARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate forecast with confidence intervals
            forecast = self.fitted_model.forecast(steps=horizon, alpha=0.05)
            conf_int = self.fitted_model.get_forecast(steps=horizon).conf_int()
            
            return {
                "predictions": forecast.tolist(),
                "prediction_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "horizon": horizon,
                "model_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error generating SARIMA predictions: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """SARIMA model has no external features."""
        return None


class XGBoostModel(BaseModel):
    """XGBoost model for time series forecasting with features."""

    def __init__(self, **xgb_params):
        super().__init__("XGBoost", "xgboost")
        self.xgb_params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            **xgb_params
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit XGBoost model."""
        try:
            import xgboost as xgb
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Fit XGBoost model
            self.model = xgb.XGBRegressor(**self.xgb_params)
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Store training metadata
            self.training_metadata = {
                "xgb_params": self.xgb_params,
                "feature_names": self.feature_names,
                "training_samples": len(y),
                "n_features": len(self.feature_names),
                "fitted_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"XGBoost model fitted with {len(self.feature_names)} features")
            
            return {
                "status": "success", 
                "message": f"XGBoost model fitted with {len(self.feature_names)} features"
            }
            
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {str(e)}")
            return {"status": "error", "message": str(e)}

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Generate XGBoost predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Ensure feature order matches training
            X_aligned = X[self.feature_names]
            
            # Generate predictions
            predictions = self.model.predict(X_aligned)
            
            # XGBoost doesn't provide prediction intervals directly
            # For now, we'll return None for intervals
            
            return {
                "predictions": predictions.tolist()[:horizon],
                "prediction_intervals": None,
                "horizon": min(horizon, len(predictions)),
                "model_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error generating XGBoost predictions: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get XGBoost feature importance."""
        if not self.is_fitted:
            return None
        
        try:
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None