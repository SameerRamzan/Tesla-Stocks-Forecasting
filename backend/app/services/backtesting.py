"""Backtesting and model evaluation service."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

from backend.app.models.forecasting import BaseModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class BacktestingService:
    """Service for backtesting and evaluating forecasting models."""

    def __init__(self):
        self.results = {}

    def walk_forward_validation(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        initial_train_size: int = 100,
        test_size: int = 1,
        step_size: int = 1,
        max_iterations: int = 50,
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation on a model.
        
        Args:
            model: Model to evaluate
            data: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature column names
            initial_train_size: Initial training window size
            test_size: Number of predictions per iteration
            step_size: How many periods to step forward
            max_iterations: Maximum number of iterations
            horizon: Forecast horizon
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting walk-forward validation for {model.name}")
        
        try:
            predictions = []
            actuals = []
            dates = []
            errors = []
            
            total_samples = len(data)
            iteration = 0
            
            for start_idx in range(initial_train_size, total_samples - test_size, step_size):
                if iteration >= max_iterations:
                    break
                
                # Define train and test sets
                train_end_idx = start_idx
                test_start_idx = start_idx
                test_end_idx = min(start_idx + test_size, total_samples)
                
                # Prepare training data
                train_data = data.iloc[:train_end_idx]
                X_train = train_data[feature_columns].dropna()
                y_train = train_data[target_column].loc[X_train.index]
                
                # Prepare test data
                test_data = data.iloc[test_start_idx:test_end_idx]
                X_test = test_data[feature_columns].dropna()
                y_test = test_data[target_column].loc[X_test.index]
                
                if len(X_train) < 10 or len(X_test) == 0:
                    continue
                
                try:
                    # Fit model
                    fit_result = model.fit(X_train, y_train)
                    if fit_result.get("status") != "success":
                        logger.warning(f"Model fitting failed at iteration {iteration}")
                        continue
                    
                    # Generate predictions
                    pred_result = model.predict(X_test, horizon=min(horizon, len(X_test)))
                    preds = pred_result["predictions"]
                    
                    # Store results
                    for i, (pred, actual) in enumerate(zip(preds, y_test.values)):
                        if not pd.isna(pred) and not pd.isna(actual):
                            predictions.append(pred)
                            actuals.append(actual)
                            dates.append(y_test.index[i] if i < len(y_test.index) else None)
                            errors.append(abs(pred - actual))
                    
                    iteration += 1
                    
                    if iteration % 10 == 0:
                        logger.info(f"Completed {iteration} iterations")
                        
                except Exception as e:
                    logger.warning(f"Error in iteration {iteration}: {str(e)}")
                    continue
            
            if not predictions:
                return {
                    "status": "error",
                    "message": "No valid predictions generated",
                    "model_name": model.name
                }
            
            # Calculate metrics
            metrics = self._calculate_metrics(actuals, predictions)
            
            return {
                "status": "success",
                "model_name": model.name,
                "iterations": iteration,
                "total_predictions": len(predictions),
                "metrics": metrics,
                "predictions": predictions,
                "actuals": actuals,
                "dates": [d.isoformat() if d else None for d in dates],
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error in walk-forward validation: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "model_name": model.name
            }

    def expanding_window_validation(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        initial_train_size: int = 100,
        test_size: int = 1,
        max_iterations: int = 50,
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Perform expanding window validation (always start from beginning).
        
        Similar to walk_forward_validation but training window always starts from index 0.
        """
        logger.info(f"Starting expanding window validation for {model.name}")
        
        try:
            predictions = []
            actuals = []
            dates = []
            errors = []
            
            total_samples = len(data)
            iteration = 0
            
            for test_start_idx in range(initial_train_size, total_samples - test_size):
                if iteration >= max_iterations:
                    break
                
                # Training data: always from beginning to test start
                train_data = data.iloc[:test_start_idx]
                X_train = train_data[feature_columns].dropna()
                y_train = train_data[target_column].loc[X_train.index]
                
                # Test data
                test_end_idx = min(test_start_idx + test_size, total_samples)
                test_data = data.iloc[test_start_idx:test_end_idx]
                X_test = test_data[feature_columns].dropna()
                y_test = test_data[target_column].loc[X_test.index]
                
                if len(X_train) < 10 or len(X_test) == 0:
                    continue
                
                try:
                    # Fit model
                    fit_result = model.fit(X_train, y_train)
                    if fit_result.get("status") != "success":
                        continue
                    
                    # Generate predictions
                    pred_result = model.predict(X_test, horizon=min(horizon, len(X_test)))
                    preds = pred_result["predictions"]
                    
                    # Store results
                    for i, (pred, actual) in enumerate(zip(preds, y_test.values)):
                        if not pd.isna(pred) and not pd.isna(actual):
                            predictions.append(pred)
                            actuals.append(actual)
                            dates.append(y_test.index[i] if i < len(y_test.index) else None)
                            errors.append(abs(pred - actual))
                    
                    iteration += 1
                    
                except Exception as e:
                    logger.warning(f"Error in iteration {iteration}: {str(e)}")
                    continue
            
            if not predictions:
                return {
                    "status": "error",
                    "message": "No valid predictions generated",
                    "model_name": model.name
                }
            
            # Calculate metrics
            metrics = self._calculate_metrics(actuals, predictions)
            
            return {
                "status": "success",
                "model_name": model.name,
                "iterations": iteration,
                "total_predictions": len(predictions),
                "metrics": metrics,
                "predictions": predictions,
                "actuals": actuals,
                "dates": [d.isoformat() if d else None for d in dates],
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error in expanding window validation: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "model_name": model.name
            }

    def _calculate_metrics(self, actuals: List[float], predictions: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        actuals_arr = np.array(actuals)
        predictions_arr = np.array(predictions)
        
        # Basic regression metrics
        mae = mean_absolute_error(actuals_arr, predictions_arr)
        mse = mean_squared_error(actuals_arr, predictions_arr)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actuals_arr - predictions_arr) / actuals_arr)) * 100
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2.0 * np.abs(predictions_arr - actuals_arr) / (np.abs(predictions_arr) + np.abs(actuals_arr))) * 100
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(actuals_arr))
        pred_direction = np.sign(np.diff(predictions_arr))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # R-squared
        ss_res = np.sum((actuals_arr - predictions_arr) ** 2)
        ss_tot = np.sum((actuals_arr - np.mean(actuals_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Median Absolute Error
        median_ae = np.median(np.abs(actuals_arr - predictions_arr))
        
        # Max error
        max_error = np.max(np.abs(actuals_arr - predictions_arr))
        
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "smape": float(smape),
            "directional_accuracy": float(directional_accuracy),
            "r2": float(r2),
            "median_absolute_error": float(median_ae),
            "max_error": float(max_error),
            "n_predictions": len(predictions)
        }

    def compare_models(self, backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model backtest results."""
        
        if not backtest_results:
            return {"status": "error", "message": "No backtest results provided"}
        
        successful_results = [r for r in backtest_results if r.get("status") == "success"]
        
        if not successful_results:
            return {"status": "error", "message": "No successful backtest results"}
        
        # Create comparison table
        comparison_data = []
        for result in successful_results:
            metrics = result["metrics"]
            comparison_data.append({
                "model_name": result["model_name"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "smape": metrics["smape"],
                "directional_accuracy": metrics["directional_accuracy"],
                "r2": metrics["r2"],
                "n_predictions": metrics["n_predictions"],
                "iterations": result["iterations"]
            })
        
        # Sort by RMSE (lower is better)
        comparison_data.sort(key=lambda x: x["rmse"])
        
        # Find best model for each metric
        best_models = {
            "mae": min(comparison_data, key=lambda x: x["mae"])["model_name"],
            "rmse": min(comparison_data, key=lambda x: x["rmse"])["model_name"],
            "mape": min(comparison_data, key=lambda x: x["mape"])["model_name"],
            "smape": min(comparison_data, key=lambda x: x["smape"])["model_name"],
            "directional_accuracy": max(comparison_data, key=lambda x: x["directional_accuracy"])["model_name"],
            "r2": max(comparison_data, key=lambda x: x["r2"])["model_name"]
        }
        
        return {
            "status": "success",
            "comparison_table": comparison_data,
            "best_models": best_models,
            "champion_model": comparison_data[0]["model_name"]  # Best RMSE
        }

    def diebold_mariano_test(
        self, 
        errors1: List[float], 
        errors2: List[float],
        h: int = 1
    ) -> Dict[str, Any]:
        """
        Perform Diebold-Mariano test for comparing forecast accuracy.
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            h: Forecast horizon (for HAC correction)
            
        Returns:
            Test results including statistic and p-value
        """
        try:
            from scipy import stats
            
            errors1_arr = np.array(errors1)
            errors2_arr = np.array(errors2)
            
            if len(errors1_arr) != len(errors2_arr):
                raise ValueError("Error arrays must have the same length")
            
            # Calculate loss differential
            d = np.abs(errors1_arr) - np.abs(errors2_arr)
            
            # Mean of loss differential
            d_bar = np.mean(d)
            
            # Standard error (simplified version)
            d_var = np.var(d, ddof=1)
            d_se = np.sqrt(d_var / len(d))
            
            # Test statistic
            dm_stat = d_bar / d_se
            
            # P-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            
            return {
                "status": "success",
                "dm_statistic": float(dm_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": "Model 1 is significantly better" if dm_stat < 0 and p_value < 0.05 
                               else "Model 2 is significantly better" if dm_stat > 0 and p_value < 0.05
                               else "No significant difference"
            }
            
        except Exception as e:
            logger.error(f"Error in Diebold-Mariano test: {str(e)}")
            return {"status": "error", "message": str(e)}

    def generate_backtest_report(self, backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        
        comparison = self.compare_models(backtest_results)
        
        if comparison.get("status") != "success":
            return comparison
        
        # Calculate additional insights
        report = {
            "summary": comparison,
            "detailed_results": backtest_results,
            "insights": {
                "total_models_tested": len([r for r in backtest_results if r.get("status") == "success"]),
                "champion_model": comparison["champion_model"],
                "performance_spread": {
                    "rmse_range": [
                        min(r["metrics"]["rmse"] for r in backtest_results if r.get("status") == "success"),
                        max(r["metrics"]["rmse"] for r in backtest_results if r.get("status") == "success")
                    ],
                    "mape_range": [
                        min(r["metrics"]["mape"] for r in backtest_results if r.get("status") == "success"),
                        max(r["metrics"]["mape"] for r in backtest_results if r.get("status") == "success")
                    ]
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report