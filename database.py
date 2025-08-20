#!/usr/bin/env python3
"""
Database operations for Tesla Stock Forecasting System

This module handles all database operations for storing user inputs 
and prediction results.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


class TeslaPredictionDB:
    """Database manager for Tesla stock prediction data."""
    
    def __init__(self, db_path: str = "tesla_predictions.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create prediction_requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    intervals_requested INTEGER NOT NULL,
                    simulation_days INTEGER NOT NULL,
                    user_ip TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    minutes_ahead INTEGER NOT NULL,
                    arima_prediction REAL NOT NULL,
                    arima_lower_bound REAL NOT NULL,
                    arima_upper_bound REAL NOT NULL,
                    sarima_prediction REAL NOT NULL,
                    sarima_lower_bound REAL NOT NULL,
                    sarima_upper_bound REAL NOT NULL,
                    confidence_width REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (request_id) REFERENCES prediction_requests (id)
                )
            """)
            
            # Create model_metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER NOT NULL,
                    arima_rmse REAL NOT NULL,
                    arima_mae REAL NOT NULL,
                    arima_aic REAL NOT NULL,
                    sarima_rmse REAL NOT NULL,
                    sarima_mae REAL NOT NULL,
                    sarima_aic REAL NOT NULL,
                    best_model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (request_id) REFERENCES prediction_requests (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_requests_timestamp 
                ON prediction_requests(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_request_id 
                ON predictions(request_id)
            """)
            
            conn.commit()
            print("✓ Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def store_prediction_request(self, intervals: int, simulation_days: int = 30, 
                               user_ip: str = None) -> int:
        """
        Store a new prediction request.
        
        Args:
            intervals: Number of intervals requested
            simulation_days: Number of days used for simulation
            user_ip: Client IP address
            
        Returns:
            ID of the created request
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO prediction_requests 
                (timestamp, intervals_requested, simulation_days, user_ip, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (now, intervals, simulation_days, user_ip, now))
            
            request_id = cursor.lastrowid
            conn.commit()
            
            print(f"✓ Stored prediction request #{request_id}")
            return request_id
    
    def store_predictions(self, request_id: int, predictions_data: Dict) -> bool:
        """
        Store prediction results for a request.
        
        Args:
            request_id: ID of the prediction request
            predictions_data: Prediction data from the predictor
            
        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            for i, timestamp in enumerate(predictions_data['timestamps']):
                minutes_ahead = (i + 1) * 15
                
                arima_pred = float(predictions_data['arima']['predictions'][i])
                arima_lower = float(predictions_data['arima']['lower_bound'][i])
                arima_upper = float(predictions_data['arima']['upper_bound'][i])
                
                sarima_pred = float(predictions_data['sarima']['predictions'][i])
                sarima_lower = float(predictions_data['sarima']['lower_bound'][i])
                sarima_upper = float(predictions_data['sarima']['upper_bound'][i])
                
                confidence_width = arima_upper - arima_lower
                
                cursor.execute("""
                    INSERT INTO predictions 
                    (request_id, prediction_timestamp, minutes_ahead, 
                     arima_prediction, arima_lower_bound, arima_upper_bound,
                     sarima_prediction, sarima_lower_bound, sarima_upper_bound,
                     confidence_width, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (request_id, timestamp.isoformat(), minutes_ahead,
                      arima_pred, arima_lower, arima_upper,
                      sarima_pred, sarima_lower, sarima_upper,
                      confidence_width, now))
            
            conn.commit()
            print(f"✓ Stored {len(predictions_data['timestamps'])} predictions for request #{request_id}")
            return True
    
    def store_model_metrics(self, request_id: int, metrics: Dict, model_info: Dict) -> bool:
        """
        Store model performance metrics.
        
        Args:
            request_id: ID of the prediction request
            metrics: Model evaluation metrics
            model_info: Model information (AIC values, etc.)
            
        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            best_model = 'sarima' if model_info['sarima_aic'] < model_info['arima_aic'] else 'arima'
            
            cursor.execute("""
                INSERT INTO model_metrics 
                (request_id, arima_rmse, arima_mae, arima_aic,
                 sarima_rmse, sarima_mae, sarima_aic, best_model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (request_id, metrics['arima']['rmse'], metrics['arima']['mae'], model_info['arima_aic'],
                  metrics['sarima']['rmse'], metrics['sarima']['mae'], model_info['sarima_aic'],
                  best_model, now))
            
            conn.commit()
            print(f"✓ Stored model metrics for request #{request_id}")
            return True
    
    def get_recent_requests(self, limit: int = 10) -> List[Dict]:
        """
        Get recent prediction requests.
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            List of recent requests
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, timestamp, intervals_requested, simulation_days, 
                       user_ip, created_at
                FROM prediction_requests 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            requests = []
            for row in cursor.fetchall():
                requests.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'intervals_requested': row['intervals_requested'],
                    'simulation_days': row['simulation_days'],
                    'user_ip': row['user_ip'],
                    'created_at': row['created_at']
                })
            
            return requests
    
    def get_predictions_for_request(self, request_id: int) -> List[Dict]:
        """
        Get all predictions for a specific request.
        
        Args:
            request_id: ID of the prediction request
            
        Returns:
            List of predictions
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT prediction_timestamp, minutes_ahead,
                       arima_prediction, arima_lower_bound, arima_upper_bound,
                       sarima_prediction, sarima_lower_bound, sarima_upper_bound,
                       confidence_width, created_at
                FROM predictions 
                WHERE request_id = ?
                ORDER BY minutes_ahead
            """, (request_id,))
            
            predictions = []
            for row in cursor.fetchall():
                predictions.append({
                    'prediction_timestamp': row['prediction_timestamp'],
                    'minutes_ahead': row['minutes_ahead'],
                    'arima': {
                        'prediction': row['arima_prediction'],
                        'lower_bound': row['arima_lower_bound'],
                        'upper_bound': row['arima_upper_bound']
                    },
                    'sarima': {
                        'prediction': row['sarima_prediction'],
                        'lower_bound': row['sarima_lower_bound'],
                        'upper_bound': row['sarima_upper_bound']
                    },
                    'confidence_width': row['confidence_width'],
                    'created_at': row['created_at']
                })
            
            return predictions
    
    def get_model_metrics_for_request(self, request_id: int) -> Optional[Dict]:
        """
        Get model metrics for a specific request.
        
        Args:
            request_id: ID of the prediction request
            
        Returns:
            Model metrics or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT arima_rmse, arima_mae, arima_aic,
                       sarima_rmse, sarima_mae, sarima_aic,
                       best_model, created_at
                FROM model_metrics 
                WHERE request_id = ?
            """, (request_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'arima': {
                        'rmse': row['arima_rmse'],
                        'mae': row['arima_mae'],
                        'aic': row['arima_aic']
                    },
                    'sarima': {
                        'rmse': row['sarima_rmse'],
                        'mae': row['sarima_mae'],
                        'aic': row['sarima_aic']
                    },
                    'best_model': row['best_model'],
                    'created_at': row['created_at']
                }
            
            return None
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Database statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count requests
            cursor.execute("SELECT COUNT(*) FROM prediction_requests")
            request_count = cursor.fetchone()[0]
            
            # Count predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cursor.fetchone()[0]
            
            # Get most recent request
            cursor.execute("""
                SELECT created_at FROM prediction_requests 
                ORDER BY created_at DESC LIMIT 1
            """)
            latest_request = cursor.fetchone()
            latest_request_time = latest_request[0] if latest_request else None
            
            # Average intervals per request
            cursor.execute("""
                SELECT AVG(intervals_requested) FROM prediction_requests
            """)
            avg_intervals = cursor.fetchone()[0] or 0
            
            return {
                'total_requests': request_count,
                'total_predictions': prediction_count,
                'latest_request': latest_request_time,
                'average_intervals_per_request': round(avg_intervals, 2),
                'database_size_mb': round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0
            }


# Test function for database functionality
def test_database():
    """Test database functionality."""
    print("🧪 Testing Tesla Prediction Database")
    print("=" * 40)
    
    # Initialize database
    db = TeslaPredictionDB("test_tesla_predictions.db")
    
    # Test storing a request
    request_id = db.store_prediction_request(4, 30, "127.0.0.1")
    
    # Test storing predictions (mock data)
    from datetime import datetime, timedelta
    
    mock_predictions = {
        'timestamps': [datetime.now() + timedelta(minutes=15 * (i + 1)) for i in range(4)],
        'arima': {
            'predictions': [100.0, 102.5, 105.0, 107.5],
            'lower_bound': [95.0, 97.5, 100.0, 102.5],
            'upper_bound': [105.0, 107.5, 110.0, 112.5]
        },
        'sarima': {
            'predictions': [101.0, 103.0, 106.0, 108.0],
            'lower_bound': [96.0, 98.0, 101.0, 103.0],
            'upper_bound': [106.0, 108.0, 111.0, 113.0]
        }
    }
    
    db.store_predictions(request_id, mock_predictions)
    
    # Test storing metrics
    mock_metrics = {
        'arima': {'rmse': 28.84, 'mae': 5.67},
        'sarima': {'rmse': 32.27, 'mae': 7.79}
    }
    
    mock_model_info = {
        'arima_aic': 1645.58,
        'sarima_aic': 1130.68
    }
    
    db.store_model_metrics(request_id, mock_metrics, mock_model_info)
    
    # Test retrieval
    requests = db.get_recent_requests(5)
    print(f"Recent requests: {len(requests)}")
    
    predictions = db.get_predictions_for_request(request_id)
    print(f"Predictions for request {request_id}: {len(predictions)}")
    
    metrics = db.get_model_metrics_for_request(request_id)
    print(f"Metrics retrieved: {metrics is not None}")
    
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Cleanup test database
    os.remove("test_tesla_predictions.db")
    print("✓ Database test completed successfully")


if __name__ == "__main__":
    test_database()