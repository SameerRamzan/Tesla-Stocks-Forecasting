#!/usr/bin/env python3
"""
Tesla 15-Minute Stock Price Prediction API Server

A simple Flask-based API server for the Tesla 15-minute stock prediction system.
Provides REST endpoints for real-time predictions.
"""

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Install with: pip install flask flask-cors")
    exit(1)

from tesla_15min_predictor import Tesla15MinPredictor
from database import TeslaPredictionDB
import json
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None
last_update = None
prediction_cache = None
db = None

def initialize_predictor():
    """Initialize the predictor with fresh data."""
    global predictor, last_update, db
    try:
        # Initialize database
        db = TeslaPredictionDB()
        
        predictor = Tesla15MinPredictor()
        predictor.load_data()
        predictor.simulate_15min_data(days=30)
        predictor.fit_arima_model()
        predictor.fit_sarima_model()
        last_update = datetime.now()
        print(f"✓ Predictor initialized at {last_update}")
        print(f"✓ Database initialized and ready")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {str(e)}")
        return False

def update_predictions():
    """Background task to update predictions periodically."""
    global prediction_cache
    while True:
        try:
            if predictor:
                prediction_cache = predictor.predict_next_intervals(n_intervals=8)
                print(f"✓ Predictions updated at {datetime.now()}")
        except Exception as e:
            print(f"✗ Failed to update predictions: {str(e)}")
        
        # Update every 5 minutes
        time.sleep(300)

@app.route('/', methods=['GET'])
def home():
    """API home endpoint."""
    return jsonify({
        'service': 'Tesla 15-Minute Stock Price Prediction API',
        'version': '1.0.0',
        'status': 'active' if predictor else 'initializing',
        'last_update': last_update.isoformat() if last_update else None,
        'endpoints': {
            '/predict': 'GET - Get current predictions',
            '/predict/{intervals}': 'GET - Get predictions for specific number of intervals',
            '/health': 'GET - Health check',
            '/metrics': 'GET - Model performance metrics',
            '/history': 'GET - Recent prediction history',
            '/history/{request_id}': 'GET - Get specific prediction details',
            '/stats': 'GET - Database statistics'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if predictor else 'initializing',
        'timestamp': datetime.now().isoformat(),
        'predictor_ready': predictor is not None,
        'last_model_update': last_update.isoformat() if last_update else None
    })

@app.route('/predict', methods=['GET'])
def get_predictions():
    """Get current stock price predictions."""
    if not predictor:
        return jsonify({'error': 'Predictor not ready'}), 503
    
    try:
        intervals = request.args.get('intervals', 4, type=int)
        intervals = min(max(intervals, 1), 24)  # Limit between 1 and 24 intervals
        
        # Get client IP for logging
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        
        # Store prediction request in database
        request_id = None
        if db:
            try:
                request_id = db.store_prediction_request(intervals, 30, client_ip)
            except Exception as e:
                print(f"Warning: Could not store request in database: {e}")
        
        predictions = predictor.predict_next_intervals(n_intervals=intervals)
        
        # Store predictions in database
        if db and request_id:
            try:
                db.store_predictions(request_id, predictions)
                
                # Get and store model metrics
                metrics = predictor.evaluate_models()
                model_info = {
                    'arima_aic': float(predictor.fitted_arima.aic),
                    'sarima_aic': float(predictor.fitted_sarima.aic)
                }
                db.store_model_metrics(request_id, metrics, model_info)
            except Exception as e:
                print(f"Warning: Could not store predictions/metrics in database: {e}")
        
        # Format response
        response_data = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(predictor.processed_data['price'].iloc[-1]),
            'prediction_horizon_minutes': intervals * 15,
            'predictions': []
        }
        
        for i, timestamp in enumerate(predictions['timestamps']):
            pred_data = {
                'timestamp': timestamp.isoformat(),
                'minutes_ahead': (i + 1) * 15,
                'arima_prediction': float(predictions['arima']['predictions'][i]),
                'arima_confidence_interval': {
                    'lower': float(predictions['arima']['lower_bound'][i]),
                    'upper': float(predictions['arima']['upper_bound'][i])
                },
                'sarima_prediction': float(predictions['sarima']['predictions'][i]),
                'sarima_confidence_interval': {
                    'lower': float(predictions['sarima']['lower_bound'][i]),
                    'upper': float(predictions['sarima']['upper_bound'][i])
                }
            }
            response_data['predictions'].append(pred_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<int:intervals>', methods=['GET'])
def get_predictions_specific(intervals):
    """Get predictions for a specific number of intervals."""
    if not predictor:
        return jsonify({'error': 'Predictor not ready'}), 503
    
    if intervals < 1 or intervals > 24:
        return jsonify({'error': 'Intervals must be between 1 and 24'}), 400
    
    try:
        predictions = predictor.predict_next_intervals(n_intervals=intervals)
        
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(predictor.processed_data['price'].iloc[-1]),
            'intervals_requested': intervals,
            'prediction_horizon_minutes': intervals * 15,
            'summary': {
                'arima_avg_prediction': float(predictions['arima']['predictions'].mean()),
                'sarima_avg_prediction': float(predictions['sarima']['predictions'].mean()),
                'trend': 'bullish' if predictions['sarima']['predictions'][-1] > predictor.processed_data['price'].iloc[-1] else 'bearish'
            },
            'detailed_predictions': []
        }
        
        for i, timestamp in enumerate(predictions['timestamps']):
            pred_data = {
                'interval': i + 1,
                'timestamp': timestamp.isoformat(),
                'minutes_ahead': (i + 1) * 15,
                'arima': float(predictions['arima']['predictions'][i]),
                'sarima': float(predictions['sarima']['predictions'][i]),
                'confidence_width': float(predictions['arima']['upper_bound'][i] - predictions['arima']['lower_bound'][i])
            }
            response_data['detailed_predictions'].append(pred_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics."""
    if not predictor:
        return jsonify({'error': 'Predictor not ready'}), 503
    
    try:
        metrics = predictor.evaluate_models()
        
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {
                'arima': {
                    'rmse': metrics['arima']['rmse'],
                    'mae': metrics['arima']['mae'],
                    'aic': float(predictor.fitted_arima.aic)
                },
                'sarima': {
                    'rmse': metrics['sarima']['rmse'],
                    'mae': metrics['sarima']['mae'],
                    'aic': float(predictor.fitted_sarima.aic)
                }
            },
            'best_model': 'sarima' if predictor.fitted_sarima.aic < predictor.fitted_arima.aic else 'arima',
            'data_summary': {
                'total_data_points': len(predictor.processed_data),
                'data_range': {
                    'start': predictor.processed_data.index[0].isoformat(),
                    'end': predictor.processed_data.index[-1].isoformat()
                },
                'price_range': {
                    'min': float(predictor.processed_data['price'].min()),
                    'max': float(predictor.processed_data['price'].max()),
                    'current': float(predictor.processed_data['price'].iloc[-1])
                }
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_prediction_history():
    """Get recent prediction history."""
    if not db:
        return jsonify({'error': 'Database not available'}), 503
    
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(max(limit, 1), 100)  # Limit between 1 and 100
        
        requests = db.get_recent_requests(limit)
        
        # Add prediction count for each request
        for req in requests:
            predictions = db.get_predictions_for_request(req['id'])
            req['prediction_count'] = len(predictions)
            
            # Add basic summary
            if predictions:
                arima_avg = sum(p['arima']['prediction'] for p in predictions) / len(predictions)
                sarima_avg = sum(p['sarima']['prediction'] for p in predictions) / len(predictions)
                req['summary'] = {
                    'arima_avg_prediction': round(arima_avg, 2),
                    'sarima_avg_prediction': round(sarima_avg, 2)
                }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'total_requests': len(requests),
            'requests': requests
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:request_id>', methods=['GET'])
def get_prediction_details(request_id):
    """Get detailed predictions for a specific request."""
    if not db:
        return jsonify({'error': 'Database not available'}), 503
    
    try:
        # Get request info
        requests = db.get_recent_requests(1000)  # Get many to find the specific one
        request_info = next((r for r in requests if r['id'] == request_id), None)
        
        if not request_info:
            return jsonify({'error': 'Request not found'}), 404
        
        # Get predictions
        predictions = db.get_predictions_for_request(request_id)
        
        # Get metrics
        metrics = db.get_model_metrics_for_request(request_id)
        
        response_data = {
            'request_info': request_info,
            'predictions': predictions,
            'metrics': metrics,
            'summary': {
                'total_predictions': len(predictions),
                'prediction_horizon_minutes': len(predictions) * 15
            }
        }
        
        if predictions:
            arima_predictions = [p['arima']['prediction'] for p in predictions]
            sarima_predictions = [p['sarima']['prediction'] for p in predictions]
            
            response_data['summary'].update({
                'arima_range': {
                    'min': min(arima_predictions),
                    'max': max(arima_predictions),
                    'avg': sum(arima_predictions) / len(arima_predictions)
                },
                'sarima_range': {
                    'min': min(sarima_predictions),
                    'max': max(sarima_predictions),
                    'avg': sum(sarima_predictions) / len(sarima_predictions)
                }
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics."""
    if not db:
        return jsonify({'error': 'Database not available'}), 503
    
    try:
        stats = db.get_database_stats()
        
        # Add system information
        stats.update({
            'predictor_status': 'ready' if predictor else 'not_ready',
            'last_model_update': last_update.isoformat() if last_update else None,
            'api_version': '1.0.0'
        })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Run the API server."""
    print("🚀 Tesla 15-Minute Stock Prediction API Server")
    print("=" * 50)
    
    # Initialize predictor
    print("Initializing prediction models...")
    if not initialize_predictor():
        print("❌ Failed to initialize predictor. Exiting.")
        return 1
    
    # Start background prediction update thread
    print("Starting background prediction updates...")
    update_thread = threading.Thread(target=update_predictions, daemon=True)
    update_thread.start()
    
    # Start the Flask server
    print("Starting API server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("\n🔗 API Endpoints:")
    print("   GET /                    - API information")
    print("   GET /health              - Health check")
    print("   GET /predict             - Get predictions (default 4 intervals)")
    print("   GET /predict?intervals=8 - Get predictions with custom intervals")
    print("   GET /predict/8           - Get predictions for specific intervals")
    print("   GET /metrics             - Model performance metrics")
    print("   GET /history             - Recent prediction history")
    print("   GET /history/{id}        - Specific prediction details")
    print("   GET /stats               - Database statistics")
    print("\n📊 Example usage:")
    print("   curl http://localhost:5000/predict")
    print("   curl http://localhost:5000/predict/6")
    print("   curl http://localhost:5000/metrics")
    print("   curl http://localhost:5000/history")
    print("   curl http://localhost:5000/stats")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())