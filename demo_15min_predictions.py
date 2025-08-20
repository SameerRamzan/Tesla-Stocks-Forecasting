#!/usr/bin/env python3
"""
Demo script for Tesla 15-minute stock price prediction system.

This script demonstrates the end-to-end functionality of predicting
Tesla stock prices at 15-minute intervals.
"""

from tesla_15min_predictor import Tesla15MinPredictor
import json
from datetime import datetime


def demo_15min_predictions():
    """
    Demonstrate the 15-minute prediction system.
    """
    print("🚀 Tesla 15-Minute Stock Price Prediction Demo")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = Tesla15MinPredictor()
    
    # Step 1: Load historical data
    print("\n📊 Step 1: Loading Tesla stock data...")
    predictor.load_data()
    
    # Step 2: Simulate 15-minute intervals (in real system, use actual intraday data)
    print("\n⏰ Step 2: Simulating 15-minute interval data...")
    print("Note: In production, this would use real intraday data from a financial API")
    predictor.simulate_15min_data(days=30)
    
    # Step 3: Check data stationarity
    print("\n📈 Step 3: Testing for stationarity...")
    stationarity = predictor.check_stationarity(predictor.processed_data['price'])
    
    # Step 4: Fit forecasting models
    print("\n🤖 Step 4: Training forecasting models...")
    print("   - Fitting ARIMA model for short-term trends...")
    predictor.fit_arima_model(order=(2, 1, 2))
    
    print("   - Fitting SARIMA model for seasonal patterns...")
    # Using 96 for daily seasonality (96 15-min intervals per day)
    predictor.fit_sarima_model(order=(2, 1, 2), seasonal_order=(1, 1, 1, 96))
    
    # Step 5: Evaluate model performance
    print("\n📊 Step 5: Evaluating model performance...")
    metrics = predictor.evaluate_models()
    
    # Step 6: Generate predictions
    print("\n🔮 Step 6: Generating 15-minute predictions...")
    # Predict next 2 hours (8 intervals of 15 minutes each)
    predictions = predictor.predict_next_intervals(n_intervals=8)
    
    # Step 7: Display detailed predictions
    print("\n📋 Detailed Prediction Results:")
    print("-" * 80)
    print(f"{'Time (EST)':<20} {'Minutes Ahead':<12} {'ARIMA':<12} {'SARIMA':<12} {'Confidence'}")
    print("-" * 80)
    
    for i, timestamp in enumerate(predictions['timestamps']):
        minutes_ahead = (i + 1) * 15
        arima_pred = predictions['arima']['predictions'][i]
        sarima_pred = predictions['sarima']['predictions'][i]
        
        # Calculate confidence based on prediction interval width
        arima_width = predictions['arima']['upper_bound'][i] - predictions['arima']['lower_bound'][i]
        confidence = max(0, min(100, 100 - (arima_width / arima_pred * 100)))
        
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
              f"{minutes_ahead:>8}min    "
              f"${arima_pred:>8.2f}    "
              f"${sarima_pred:>8.2f}    "
              f"{confidence:>6.1f}%")
    
    # Step 8: Generate visualization
    print("\n📊 Step 8: Generating prediction chart...")
    predictor.plot_predictions(show_last_hours=6)
    
    # Step 9: Export results
    print("\n💾 Step 9: Exporting prediction results...")
    export_file = predictor.export_predictions("demo_predictions.json")
    
    # Step 10: Summary
    print("\n✅ Demo Summary:")
    print("-" * 40)
    print(f"Data Points Processed: {len(predictor.processed_data)}")
    print(f"Prediction Horizon: 2 hours (8 intervals)")
    print(f"ARIMA Model AIC: {predictor.fitted_arima.aic:.2f}")
    print(f"SARIMA Model AIC: {predictor.fitted_sarima.aic:.2f}")
    print(f"Best Model: {'SARIMA' if predictor.fitted_sarima.aic < predictor.fitted_arima.aic else 'ARIMA'}")
    
    # Price trend analysis
    current_price = predictor.processed_data['price'].iloc[-1]
    predicted_price_1hr = predictions['sarima']['predictions'][3]  # 4th interval = 1 hour
    trend = "📈 BULLISH" if predicted_price_1hr > current_price else "📉 BEARISH"
    change_pct = ((predicted_price_1hr - current_price) / current_price) * 100
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"1-Hour Prediction: ${predicted_price_1hr:.2f}")
    print(f"Expected Change: {change_pct:+.2f}% {trend}")
    
    print(f"\n📁 Files Generated:")
    print(f"   - tesla_15min_predictions.png (chart)")
    print(f"   - demo_predictions.json (data)")
    
    print(f"\n🎯 Next Steps for Production:")
    print("   1. Integrate with real-time market data API")
    print("   2. Implement automated retraining pipeline")
    print("   3. Add risk management and position sizing")
    print("   4. Deploy as web service or trading bot")
    print("   5. Add real-time alerts and notifications")
    
    return predictions


def analyze_prediction_accuracy():
    """
    Analyze the accuracy of predictions (demo function).
    """
    print("\n🎯 Prediction Accuracy Analysis")
    print("-" * 40)
    
    # Load the exported predictions
    try:
        with open('/home/runner/work/Tesla-Stocks-Forecasting/Tesla-Stocks-Forecasting/demo_predictions.json', 'r') as f:
            data = json.load(f)
        
        print("Prediction Details:")
        for pred in data['predictions']:
            interval = pred['interval_minutes_ahead']
            arima_pred = pred['arima_prediction']
            sarima_pred = pred['sarima_prediction']
            confidence_width = pred['arima_upper_bound'] - pred['arima_lower_bound']
            
            print(f"  {interval:2d}min: ARIMA ${arima_pred:.2f}, "
                  f"SARIMA ${sarima_pred:.2f}, "
                  f"Uncertainty: ±${confidence_width/2:.2f}")
            
    except FileNotFoundError:
        print("No prediction file found. Run the demo first.")


if __name__ == "__main__":
    try:
        # Run the main demo
        predictions = demo_15min_predictions()
        
        # Analyze accuracy
        analyze_prediction_accuracy()
        
        print("\n🎉 Demo completed successfully!")
        print("The 15-minute prediction system is ready for deployment.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Please check the data file and dependencies.")