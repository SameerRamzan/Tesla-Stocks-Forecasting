#!/usr/bin/env python3
"""
Tesla 15-Minute Stock Price Prediction System

This module provides an end-to-end system for predicting Tesla stock prices
at 15-minute intervals using time series forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import argparse
import json

warnings.filterwarnings("ignore")


class Tesla15MinPredictor:
    """
    A class for predicting Tesla stock prices at 15-minute intervals.
    
    This system can work with both daily data (for demonstration) and 
    high-frequency intraday data when available.
    """
    
    def __init__(self, data_source: str = None):
        """
        Initialize the predictor.
        
        Args:
            data_source: Path to the data file (CSV format)
        """
        self.data_source = data_source
        self.raw_data = None
        self.processed_data = None
        self.model_arima = None
        self.model_sarima = None
        self.forecast_results = None
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load stock data from CSV file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame with datetime index
        """
        if data_path:
            self.data_source = data_path
            
        if not self.data_source:
            self.data_source = "Tesla Stock Dataset.csv"
            
        try:
            df = pd.read_csv(self.data_source)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            self.raw_data = df
            print(f"✓ Successfully loaded {len(df)} records from {self.data_source}")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def simulate_15min_data(self, days: int = 30) -> pd.DataFrame:
        """
        Simulate 15-minute interval data from daily data for demonstration.
        
        In a real implementation, this would be replaced with actual 
        15-minute intraday data from a financial data provider.
        
        Args:
            days: Number of recent days to simulate
            
        Returns:
            DataFrame with 15-minute intervals
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get recent data
        recent_data = self.raw_data.tail(days).copy()
        
        # Create 15-minute intervals for each trading day
        # Assuming trading hours: 9:30 AM to 4:00 PM EST (6.5 hours = 26 15-min intervals)
        trading_intervals = 26
        
        simulated_data = []
        
        for date, row in recent_data.iterrows():
            # Generate trading session times for this day
            start_time = date.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Generate price movements within the day
            daily_return = (row['Close'] - row['Open']) / row['Open']
            volatility = (row['High'] - row['Low']) / row['Open']
            
            # Simulate random walk with drift for intraday prices
            np.random.seed(int(date.timestamp()) % 2**32)  # Reproducible randomness
            
            current_price = row['Open']
            interval_returns = np.random.normal(
                daily_return / trading_intervals,  # Mean return per interval
                volatility / np.sqrt(trading_intervals),  # Volatility scaling
                trading_intervals
            )
            
            for i in range(trading_intervals):
                timestamp = start_time + timedelta(minutes=15 * i)
                current_price *= (1 + interval_returns[i])
                
                simulated_data.append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'volume': row['Volume'] / trading_intervals  # Distribute volume
                })
        
        sim_df = pd.DataFrame(simulated_data)
        sim_df.set_index('timestamp', inplace=True)
        
        self.processed_data = sim_df
        print(f"✓ Simulated {len(sim_df)} 15-minute intervals from {days} days of data")
        return sim_df
    
    def prepare_real_15min_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare real 15-minute interval data for modeling.
        
        This method would be used when actual 15-minute data is available.
        
        Args:
            data: DataFrame with 15-minute interval data
            
        Returns:
            Processed DataFrame ready for modeling
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Handle missing values
        data = data.fillna(method='forward').fillna(method='backward')
        
        # Filter to trading hours if needed
        trading_hours = data.between_time('09:30', '16:00')
        
        self.processed_data = trading_hours
        return trading_hours
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Check if the time series is stationary using ADF test.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"P-value: {result[1]:.6f}")
        print(f"Critical Values: {result[4]}")
        print(f"Series is {'stationary' if stationarity_result['is_stationary'] else 'non-stationary'}")
        
        return stationarity_result
    
    def fit_arima_model(self, order: Tuple[int, int, int] = (2, 1, 2)) -> None:
        """
        Fit ARIMA model to the data.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call simulate_15min_data() or prepare_real_15min_data() first.")
        
        price_series = self.processed_data['price']
        
        print(f"Fitting ARIMA{order} model...")
        self.model_arima = ARIMA(price_series, order=order)
        self.fitted_arima = self.model_arima.fit()
        
        print("✓ ARIMA model fitted successfully")
        print(f"AIC: {self.fitted_arima.aic:.2f}")
        
    def fit_sarima_model(self, order: Tuple[int, int, int] = (2, 1, 2), 
                        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 96)) -> None:
        """
        Fit SARIMA model to the data.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
                          s=96 for daily seasonality in 15-min data (24*4=96 intervals per day)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        price_series = self.processed_data['price']
        
        print(f"Fitting SARIMA{order}x{seasonal_order} model...")
        self.model_sarima = SARIMAX(price_series, order=order, seasonal_order=seasonal_order)
        self.fitted_sarima = self.model_sarima.fit()
        
        print("✓ SARIMA model fitted successfully")
        print(f"AIC: {self.fitted_sarima.aic:.2f}")
    
    def predict_next_intervals(self, n_intervals: int = 4) -> Dict:
        """
        Predict the next n 15-minute intervals.
        
        Args:
            n_intervals: Number of 15-minute intervals to predict (default: 4 = 1 hour)
            
        Returns:
            Dictionary with predictions from both models
        """
        if self.fitted_arima is None or self.fitted_sarima is None:
            raise ValueError("Models not fitted. Call fit_arima_model() and fit_sarima_model() first.")
        
        # Generate future timestamps
        last_timestamp = self.processed_data.index[-1]
        future_timestamps = [
            last_timestamp + timedelta(minutes=15 * (i + 1)) 
            for i in range(n_intervals)
        ]
        
        # ARIMA predictions
        arima_forecast = self.fitted_arima.get_forecast(steps=n_intervals)
        arima_pred = arima_forecast.predicted_mean
        arima_conf_int = arima_forecast.conf_int()
        
        # SARIMA predictions
        sarima_forecast = self.fitted_sarima.get_forecast(steps=n_intervals)
        sarima_pred = sarima_forecast.predicted_mean
        sarima_conf_int = sarima_forecast.conf_int()
        
        self.forecast_results = {
            'timestamps': future_timestamps,
            'arima': {
                'predictions': arima_pred.values,
                'lower_bound': arima_conf_int.iloc[:, 0].values,
                'upper_bound': arima_conf_int.iloc[:, 1].values
            },
            'sarima': {
                'predictions': sarima_pred.values,
                'lower_bound': sarima_conf_int.iloc[:, 0].values,
                'upper_bound': sarima_conf_int.iloc[:, 1].values
            }
        }
        
        print(f"✓ Generated predictions for next {n_intervals} intervals (next {n_intervals * 15} minutes)")
        return self.forecast_results
    
    def evaluate_models(self) -> Dict:
        """
        Evaluate model performance on historical data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.fitted_arima is None or self.fitted_sarima is None:
            raise ValueError("Models not fitted.")
        
        actual = self.processed_data['price']
        
        # Get fitted values
        arima_fitted = self.fitted_arima.fittedvalues
        sarima_fitted = self.fitted_sarima.fittedvalues
        
        # Calculate metrics
        arima_rmse = sqrt(mean_squared_error(actual, arima_fitted))
        sarima_rmse = sqrt(mean_squared_error(actual, sarima_fitted))
        
        arima_mae = mean_absolute_error(actual, arima_fitted)
        sarima_mae = mean_absolute_error(actual, sarima_fitted)
        
        metrics = {
            'arima': {
                'rmse': arima_rmse,
                'mae': arima_mae
            },
            'sarima': {
                'rmse': sarima_rmse,
                'mae': sarima_mae
            }
        }
        
        print("Model Evaluation Results:")
        print(f"ARIMA  - RMSE: ${arima_rmse:.2f}, MAE: ${arima_mae:.2f}")
        print(f"SARIMA - RMSE: ${sarima_rmse:.2f}, MAE: ${sarima_mae:.2f}")
        
        return metrics
    
    def plot_predictions(self, show_last_hours: int = 24):
        """
        Plot the predictions along with historical data.
        
        Args:
            show_last_hours: Number of hours of historical data to show
        """
        if self.forecast_results is None:
            raise ValueError("No predictions available. Call predict_next_intervals() first.")
        
        # Get recent data for plotting
        cutoff_time = self.processed_data.index[-1] - timedelta(hours=show_last_hours)
        recent_data = self.processed_data.loc[self.processed_data.index >= cutoff_time]
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(recent_data.index, recent_data['price'], 
                label='Historical Prices', color='blue', linewidth=2)
        
        # Plot predictions
        timestamps = self.forecast_results['timestamps']
        
        # ARIMA predictions
        arima_pred = self.forecast_results['arima']['predictions']
        plt.plot(timestamps, arima_pred, 
                label='ARIMA Predictions', color='red', marker='o', linewidth=2)
        plt.fill_between(timestamps, 
                        self.forecast_results['arima']['lower_bound'],
                        self.forecast_results['arima']['upper_bound'],
                        alpha=0.3, color='red')
        
        # SARIMA predictions
        sarima_pred = self.forecast_results['sarima']['predictions']
        plt.plot(timestamps, sarima_pred, 
                label='SARIMA Predictions', color='green', marker='s', linewidth=2)
        plt.fill_between(timestamps,
                        self.forecast_results['sarima']['lower_bound'],
                        self.forecast_results['sarima']['upper_bound'],
                        alpha=0.3, color='green')
        
        plt.title('Tesla Stock Price: 15-Minute Predictions', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('/home/runner/work/Tesla-Stocks-Forecasting/Tesla-Stocks-Forecasting/tesla_15min_predictions.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Plot saved as 'tesla_15min_predictions.png'")
    
    def export_predictions(self, filename: str = None) -> str:
        """
        Export predictions to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if self.forecast_results is None:
            raise ValueError("No predictions available.")
        
        if filename is None:
            filename = f"tesla_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        export_data = {
            'prediction_time': datetime.now().isoformat(),
            'model_info': {
                'arima_aic': float(self.fitted_arima.aic),
                'sarima_aic': float(self.fitted_sarima.aic)
            },
            'predictions': []
        }
        
        for i, timestamp in enumerate(self.forecast_results['timestamps']):
            prediction = {
                'timestamp': timestamp.isoformat(),
                'interval_minutes_ahead': (i + 1) * 15,
                'arima_prediction': float(self.forecast_results['arima']['predictions'][i]),
                'arima_lower_bound': float(self.forecast_results['arima']['lower_bound'][i]),
                'arima_upper_bound': float(self.forecast_results['arima']['upper_bound'][i]),
                'sarima_prediction': float(self.forecast_results['sarima']['predictions'][i]),
                'sarima_lower_bound': float(self.forecast_results['sarima']['lower_bound'][i]),
                'sarima_upper_bound': float(self.forecast_results['sarima']['upper_bound'][i])
            }
            export_data['predictions'].append(prediction)
        
        filepath = f"/home/runner/work/Tesla-Stocks-Forecasting/Tesla-Stocks-Forecasting/{filename}"
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Predictions exported to {filename}")
        return filepath


def main():
    """
    Command-line interface for the Tesla 15-minute predictor.
    """
    parser = argparse.ArgumentParser(description='Tesla 15-Minute Stock Price Predictor')
    parser.add_argument('--data', type=str, default='Tesla Stock Dataset.csv',
                       help='Path to the stock data CSV file')
    parser.add_argument('--intervals', type=int, default=4,
                       help='Number of 15-minute intervals to predict')
    parser.add_argument('--simulate-days', type=int, default=30,
                       help='Number of recent days to use for simulation')
    parser.add_argument('--export', type=str, default=None,
                       help='Export predictions to JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and show prediction plot')
    
    args = parser.parse_args()
    
    print("Tesla 15-Minute Stock Price Prediction System")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = Tesla15MinPredictor(args.data)
        
        # Load data
        predictor.load_data()
        
        # Simulate 15-minute data (replace with real data when available)
        predictor.simulate_15min_data(days=args.simulate_days)
        
        # Check stationarity
        print("\nStationarity Test:")
        predictor.check_stationarity(predictor.processed_data['price'])
        
        # Fit models
        print("\nFitting Models:")
        predictor.fit_arima_model()
        predictor.fit_sarima_model()
        
        # Evaluate models
        print("\nModel Evaluation:")
        predictor.evaluate_models()
        
        # Make predictions
        print(f"\nGenerating Predictions for next {args.intervals} intervals:")
        predictions = predictor.predict_next_intervals(args.intervals)
        
        # Display predictions
        print("\nPrediction Results:")
        for i, timestamp in enumerate(predictions['timestamps']):
            minutes_ahead = (i + 1) * 15
            arima_pred = predictions['arima']['predictions'][i]
            sarima_pred = predictions['sarima']['predictions'][i]
            
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} (+{minutes_ahead:2d}min): "
                  f"ARIMA: ${arima_pred:.2f}, SARIMA: ${sarima_pred:.2f}")
        
        # Export predictions if requested
        if args.export:
            predictor.export_predictions(args.export)
        
        # Generate plot if requested
        if args.plot:
            predictor.plot_predictions()
        
        print("\n✓ Prediction system completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())