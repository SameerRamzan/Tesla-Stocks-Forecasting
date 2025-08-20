#!/usr/bin/env python3
"""
Test script for Tesla 15-minute stock prediction system.

This script validates that all components of the system work correctly.
"""

import sys
import os
import json
from datetime import datetime
import unittest

# Add the current directory to Python path
sys.path.append('/home/runner/work/Tesla-Stocks-Forecasting/Tesla-Stocks-Forecasting')

from tesla_15min_predictor import Tesla15MinPredictor


class TestTesla15MinPredictor(unittest.TestCase):
    """Test cases for the Tesla 15-minute predictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = Tesla15MinPredictor()
        
    def test_data_loading(self):
        """Test data loading functionality."""
        print("Testing data loading...")
        data = self.predictor.load_data()
        
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 1000)  # Should have substantial data
        self.assertIn('Close', data.columns)
        print("✓ Data loading test passed")
        
    def test_simulation(self):
        """Test 15-minute data simulation."""
        print("Testing 15-minute data simulation...")
        self.predictor.load_data()
        sim_data = self.predictor.simulate_15min_data(days=5)
        
        self.assertIsNotNone(sim_data)
        self.assertGreater(len(sim_data), 100)  # Should have multiple intervals
        self.assertIn('price', sim_data.columns)
        print("✓ Simulation test passed")
        
    def test_model_fitting(self):
        """Test model fitting."""
        print("Testing model fitting...")
        self.predictor.load_data()
        self.predictor.simulate_15min_data(days=10)
        
        # Test ARIMA
        self.predictor.fit_arima_model()
        self.assertIsNotNone(self.predictor.fitted_arima)
        
        # Test SARIMA
        self.predictor.fit_sarima_model()
        self.assertIsNotNone(self.predictor.fitted_sarima)
        
        print("✓ Model fitting test passed")
        
    def test_predictions(self):
        """Test prediction generation."""
        print("Testing prediction generation...")
        self.predictor.load_data()
        self.predictor.simulate_15min_data(days=10)
        self.predictor.fit_arima_model()
        self.predictor.fit_sarima_model()
        
        predictions = self.predictor.predict_next_intervals(n_intervals=4)
        
        self.assertIsNotNone(predictions)
        self.assertIn('timestamps', predictions)
        self.assertIn('arima', predictions)
        self.assertIn('sarima', predictions)
        self.assertEqual(len(predictions['timestamps']), 4)
        
        print("✓ Prediction generation test passed")
        
    def test_evaluation(self):
        """Test model evaluation."""
        print("Testing model evaluation...")
        self.predictor.load_data()
        self.predictor.simulate_15min_data(days=10)
        self.predictor.fit_arima_model()
        self.predictor.fit_sarima_model()
        
        metrics = self.predictor.evaluate_models()
        
        self.assertIsNotNone(metrics)
        self.assertIn('arima', metrics)
        self.assertIn('sarima', metrics)
        self.assertIn('rmse', metrics['arima'])
        self.assertIn('mae', metrics['arima'])
        
        print("✓ Model evaluation test passed")


def test_api_imports():
    """Test that API dependencies can be imported."""
    print("Testing API dependencies...")
    try:
        import flask
        import flask_cors
        print("✓ Flask dependencies available")
        return True
    except ImportError as e:
        print(f"⚠ Flask dependencies not available: {e}")
        print("  Install with: pip install flask flask-cors")
        return False


def test_file_generation():
    """Test that prediction files are generated correctly."""
    print("Testing file generation...")
    
    # Check if prediction files exist
    files_to_check = [
        'tesla_15min_predictions.png',
        'demo_predictions.json',
        'tesla_predictions.json'
    ]
    
    all_exist = True
    for filename in files_to_check:
        filepath = f'/home/runner/work/Tesla-Stocks-Forecasting/Tesla-Stocks-Forecasting/{filename}'
        if os.path.exists(filepath):
            print(f"✓ {filename} exists")
            
            # For JSON files, validate they contain valid JSON
            if filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    print(f"  - Valid JSON with {len(data.get('predictions', []))} predictions")
                except Exception as e:
                    print(f"  - Invalid JSON: {e}")
                    all_exist = False
        else:
            print(f"✗ {filename} missing")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("🧪 Tesla 15-Minute Prediction System Tests")
    print("=" * 50)
    
    # Test API dependencies
    api_available = test_api_imports()
    
    # Test file generation
    print("\n" + "=" * 50)
    files_ok = test_file_generation()
    
    # Run unit tests
    print("\n" + "=" * 50)
    print("Running unit tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTesla15MinPredictor)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Unit Tests: {'✓ PASSED' if result.wasSuccessful() else '✗ FAILED'}")
    print(f"API Dependencies: {'✓ AVAILABLE' if api_available else '⚠ MISSING'}")
    print(f"Output Files: {'✓ GENERATED' if files_ok else '✗ MISSING'}")
    
    if result.wasSuccessful() and files_ok:
        print("\n🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())