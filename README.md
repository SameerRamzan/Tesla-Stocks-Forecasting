# Tesla Stocks Forecasting Using ARIMA & SARIMA

## Project Description

This repository contains a comprehensive time-series analysis and forecasting system for Tesla, Inc. (TSLA) stock prices. The project now includes both daily forecasting capabilities using the original Jupyter notebook and **a new end-to-end system for 15-minute stock price predictions**.

### 🆕 New Feature: 15-Minute Stock Price Prediction System

The repository now includes a complete end-to-end system for predicting Tesla stock prices at 15-minute intervals, featuring:

- **Real-time prediction capabilities** using ARIMA and SARIMA models
- **Command-line interface** for easy predictions
- **REST API server** for integration with other systems
- **Automated model retraining** and evaluation
- **Visualization and export** of predictions
- **Comprehensive error handling** and validation

## Files in this Repository

### Original Files
*   **`Tesla Stock Dataset.csv`**: A CSV file containing daily historical stock price data for Tesla (TSLA) from June 29, 2010, to January 3, 2025.
*   **`Tesla_stock_Forecasting___ARIMA_SARIMA.ipynb`**: A Jupyter Notebook that includes the Python code for analyzing the stock data and implementing the ARIMA and SARIMA forecasting models for daily predictions.

### 🆕 New 15-Minute Prediction System Files
*   **`tesla_15min_predictor.py`**: Main prediction engine for 15-minute stock price forecasting
*   **`demo_15min_predictions.py`**: Interactive demo script showcasing the 15-minute prediction system
*   **`api_server.py`**: REST API server for real-time predictions via HTTP endpoints
*   **`requirements.txt`**: Python dependencies for the entire system
*   **`README.md`**: This file, providing comprehensive documentation

## 🚀 Quick Start - 15-Minute Predictions

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SameerRamzan/Tesla-Stocks-Forecasting.git
   cd Tesla-Stocks-Forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage Options

#### Option 1: Command Line Interface
```bash
# Basic prediction (next 4 intervals = 1 hour)
python tesla_15min_predictor.py

# Custom prediction (next 8 intervals = 2 hours) with visualization
python tesla_15min_predictor.py --intervals 8 --plot --export predictions.json

# Help and options
python tesla_15min_predictor.py --help
```

#### Option 2: Interactive Demo
```bash
# Run the comprehensive demo
python demo_15min_predictions.py
```

#### Option 3: REST API Server
```bash
# Start the API server
python api_server.py
```

Then access the API endpoints:
- `GET http://localhost:5000/` - API information
- `GET http://localhost:5000/predict` - Get predictions (default 4 intervals)
- `GET http://localhost:5000/predict/8` - Get predictions for 8 intervals
- `GET http://localhost:5000/metrics` - Model performance metrics

### Example API Usage
```bash
# Get next 6 15-minute predictions
curl http://localhost:5000/predict/6

# Get model performance metrics
curl http://localhost:5000/metrics
```

## 📊 System Architecture

### 15-Minute Prediction Pipeline

1. **Data Loading**: Load historical Tesla stock data
2. **Data Processing**: Create 15-minute interval data (simulated from daily data for demo)
3. **Model Training**: Fit ARIMA and SARIMA models optimized for high-frequency trading
4. **Prediction Generation**: Generate forecasts for next N 15-minute intervals
5. **Evaluation & Visualization**: Assess model performance and create charts
6. **Export & API**: Save results and serve via REST API

### Model Specifications

#### For 15-Minute Predictions:
- **ARIMA Model**: (2, 1, 2) - Optimized for short-term price movements
- **SARIMA Model**: (2, 1, 2) x (1, 1, 1, 96) - Accounts for daily seasonality (96 15-min intervals per day)

#### For Daily Predictions (Original):
- **ARIMA Model**: (1, 1, 1) - As implemented in the original notebook
- **SARIMA Model**: (2, 1, 2) x (1, 1, 1, 12) - Monthly seasonality

## Dataset

The `Tesla Stock Dataset.csv` includes the following columns:

*   **`Date`**: The trading date (YYYY-MM-DD).
*   **`Open`**: The opening stock price on the given date.
*   **`High`**: The highest stock price during the trading day.
*   **`Low`**: The lowest stock price during the trading day.
*   **`Close`**: The closing stock price for the day (primary target for daily forecasting).
*   **`Adj Close`**: The adjusted closing price, corrected for dividends and stock splits.
*   **`Volume`**: The number of shares traded during the day.

**Note**: For 15-minute predictions, the system simulates intraday data from the daily data. In a production environment, this would be replaced with real 15-minute interval data from financial data providers like Alpha Vantage, Yahoo Finance, or Bloomberg API.

## 🎯 15-Minute Prediction Features

### Core Capabilities
- **High-Frequency Forecasting**: Predict Tesla stock prices for the next 15, 30, 45, 60+ minute intervals
- **Multiple Models**: Compare ARIMA and SARIMA model predictions
- **Confidence Intervals**: Get prediction uncertainty bounds
- **Real-Time Processing**: Fast prediction generation suitable for live trading
- **Automated Retraining**: Models can be retrained with new data

### Outputs Generated
- **Price Predictions**: Specific price forecasts for each 15-minute interval
- **Trend Analysis**: Bullish/bearish trend identification
- **Confidence Metrics**: Statistical confidence in predictions
- **Visualization**: Charts showing historical data and future predictions
- **JSON Export**: Machine-readable prediction data for integration

### Example Output
```
Tesla 15-Minute Predictions:
2025-01-03 16:00 (+15min): ARIMA: $393.01, SARIMA: $394.08
2025-01-03 16:15 (+30min): ARIMA: $393.31, SARIMA: $397.60
2025-01-03 16:30 (+45min): ARIMA: $393.26, SARIMA: $394.99
Expected 1-Hour Change: +0.29% 📈 BULLISH
```

## 🔧 Technical Details

### Dependencies
The system uses the following Python libraries:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **statsmodels**: Time series analysis (ARIMA/SARIMA)
- **scikit-learn**: Model evaluation metrics
- **flask**: REST API server (optional)

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Performance Metrics
The system provides comprehensive model evaluation:
- **RMSE (Root Mean Square Error)**: Prediction accuracy measure
- **MAE (Mean Absolute Error)**: Average prediction error
- **AIC (Akaike Information Criterion)**: Model quality comparison
- **Confidence Intervals**: Prediction uncertainty bounds

### API Endpoints
When running the API server (`python api_server.py`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | System health check |
| `/predict` | GET | Get predictions (default 4 intervals) |
| `/predict/{n}` | GET | Get predictions for n intervals |
| `/predict?intervals=n` | GET | Get predictions with query parameter |
| `/metrics` | GET | Model performance and evaluation metrics |

### Sample API Response
```json
{
  "timestamp": "2025-01-03T16:00:00",
  "current_price": 393.31,
  "prediction_horizon_minutes": 60,
  "predictions": [
    {
      "timestamp": "2025-01-03T16:15:00",
      "minutes_ahead": 15,
      "arima_prediction": 393.01,
      "sarima_prediction": 394.08,
      "confidence_width": 19.96
    }
  ]
}
```

## How to Use

### For 15-Minute Stock Predictions (New System)

1.  **Prerequisites**: Ensure you have Python 3.7+ installed.
2.  **Install Dependencies**: Run `pip install -r requirements.txt`
3.  **Quick Start**: Run `python demo_15min_predictions.py` for an interactive demonstration
4.  **Command Line**: Use `python tesla_15min_predictor.py --help` for all options
5.  **API Server**: Run `python api_server.py` to start the REST API service

**Example Commands:**
```bash
# Get next 8 intervals (2 hours) with visualization
python tesla_15min_predictor.py --intervals 8 --plot

# Export predictions to JSON
python tesla_15min_predictor.py --export my_predictions.json

# Start API server for integration
python api_server.py
```

### For Daily Stock Predictions (Original System)

1.  **Prerequisites**: Ensure you have Python installed, along with Jupyter Notebook or JupyterLab. You will also need the libraries listed in the "Dependencies" section.
2.  **Download Files**: Clone or download all files from this repository.
3.  **Open Notebook**: Launch Jupyter Notebook/Lab and open `Tesla_stock_Forecasting___ARIMA_SARIMA.ipynb`.
4.  **Run Cells**: Execute the cells in the notebook sequentially. The notebook will:
    *   Load the dataset.
    *   Preprocess and visualize the data.
    *   Perform stationarity tests.
    *   Build, fit, and evaluate ARIMA and SARIMA models.
    *   Generate and plot a 30-day forecast for Tesla's stock closing price.

## 🚀 Production Deployment

For deploying the 15-minute prediction system in production:

### 1. Real-Time Data Integration
Replace the simulation with real market data:
```python
# Example integration with Alpha Vantage
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries

def get_real_15min_data():
    ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol='TSLA', interval='15min')
    return data
```

### 2. Automated Retraining
Set up scheduled model retraining:
```bash
# Add to crontab for retraining every hour
0 * * * * /usr/bin/python /path/to/retrain_models.py
```

### 3. Monitoring and Alerts
Implement performance monitoring:
- Track prediction accuracy over time
- Set up alerts for unusual market conditions
- Monitor API response times and errors

### 4. Scaling Considerations
- Use Redis for caching predictions
- Deploy on cloud platforms (AWS, GCP, Azure)
- Implement load balancing for high traffic
- Use Docker for containerization

## 🛠 Customization

### Model Parameters
You can customize the models by modifying parameters in `tesla_15min_predictor.py`:

```python
# ARIMA configuration
predictor.fit_arima_model(order=(p, d, q))

# SARIMA configuration  
predictor.fit_sarima_model(order=(p, d, q), seasonal_order=(P, D, Q, s))
```

### Prediction Intervals
Adjust the prediction horizon:
```python
# Predict next 12 intervals (3 hours)
predictions = predictor.predict_next_intervals(n_intervals=12)
```

### Trading Hours
Modify trading session times in the simulation:
```python
# Custom trading hours (e.g., extended hours)
start_time = date.replace(hour=4, minute=0)  # 4 AM start
trading_intervals = 48  # 12 hours * 4 intervals per hour
```

## Model Information

### 15-Minute Prediction Models (New System)

**ARIMA Model for High-Frequency Trading:**
- Parameters (p, d, q): (2, 1, 2) - Optimized for short-term price movements
- Designed for rapid 15-minute interval predictions
- Captures short-term autoregressive patterns in stock prices

**SARIMA Model for Intraday Patterns:**
- Parameters (p, d, q): (2, 1, 2)
- Seasonal Parameters (P, D, Q, s): (1, 1, 1, 96)
- s=96 represents daily seasonality (96 15-minute intervals per trading day)
- Accounts for intraday trading patterns and market opening/closing effects

### Daily Prediction Models (Original System)

The notebook implements two time-series forecasting models:

1.  **ARIMA (Autoregressive Integrated Moving Average)**:
    *   Parameters (p, d, q): (1, 1, 1) as used in the notebook.
    *   The model is fitted to the 'Close' price after differencing to achieve stationarity.

2.  **SARIMA (Seasonal ARIMA)**:
    *   Parameters (p, d, q): (2, 1, 2)
    *   Seasonal Parameters (P, D, Q, s): (1, 1, 1, 12)
    *   This model accounts for seasonality in the time series, with a seasonal period of 12 (likely representing months if the data were aggregated monthly, though applied to daily data here with a 365-day decomposition period for visualization).

## Results/Output

### 15-Minute Prediction System Output

**Command Line Execution:**
```bash
$ python tesla_15min_predictor.py --intervals 4 --plot
✓ Successfully loaded 3654 records from Tesla Stock Dataset.csv
✓ Simulated 780 15-minute intervals from 30 days of data
✓ ARIMA model fitted successfully (AIC: 4756.53)
✓ SARIMA model fitted successfully (AIC: 4378.23)

Prediction Results:
2025-01-03 16:00 (+15min): ARIMA: $393.01, SARIMA: $394.08
2025-01-03 16:15 (+30min): ARIMA: $393.31, SARIMA: $397.60
2025-01-03 16:30 (+45min): ARIMA: $393.26, SARIMA: $394.99
2025-01-03 16:45 (+60min): ARIMA: $393.19, SARIMA: $394.46
```

**Generated Files:**
- **Visualization**: `tesla_15min_predictions.png` - Chart showing historical data and predictions
- **Data Export**: `tesla_predictions.json` - Machine-readable prediction results
- **Performance Metrics**: RMSE, MAE, and AIC scores for model evaluation

**API Server Output:**
When running `python api_server.py`, provides REST endpoints for:
- Real-time predictions at `/predict`
- Model performance metrics at `/metrics`
- Health monitoring at `/health`

### Daily Prediction System Output (Original)

Executing the Jupyter Notebook will produce:

*   **Descriptive statistics and information** about the dataset.
*   **Visualizations**:
    *   Tesla stock closing price over time.
    *   Seasonal decomposition of the time series (trend, seasonality, residuals).
    *   Differenced time series plot.
    *   ACF and PACF plots for original, differenced, and residual data.
    *   Plots of actual vs. fitted values for both ARIMA and SARIMA models.
    *   A 30-day forecast plot for the SARIMA model, including confidence intervals.
*   **Model Summaries**: Detailed statistical summaries for both ARIMA and SARIMAX models.
*   **Stationarity Test Results**: Output from the Augmented Dickey-Fuller (ADF) test.
*   **Performance Metrics**: Root Mean Squared Error (RMSE) for both ARIMA and SARIMA models to evaluate their forecasting accuracy.

## ⚠️ Important Notes

### Limitations and Disclaimers

1. **Simulated Data**: The 15-minute system currently uses simulated intraday data derived from daily prices. For production use, integrate with real-time market data providers.

2. **Market Risk**: Stock price predictions are inherently uncertain. This system is for educational and research purposes. Always conduct thorough backtesting before any trading decisions.

3. **Model Assumptions**: ARIMA and SARIMA models assume certain statistical properties of the data. Market conditions can change rapidly, affecting model performance.

4. **Regulatory Compliance**: Ensure compliance with financial regulations if using for actual trading. Consider implementing proper risk management and position sizing.

### Best Practices for Production Use

1. **Real-Time Data**: Replace simulation with live market data feeds
2. **Model Validation**: Implement continuous backtesting and validation
3. **Risk Management**: Add stop-loss and position sizing logic
4. **Performance Monitoring**: Track prediction accuracy over time
5. **Fallback Systems**: Implement redundancy for critical trading systems

## 🤝 Contributing

We welcome contributions to improve the prediction system:

1. **Real-Time Data Integration**: Add connectors for popular financial data APIs
2. **Advanced Models**: Implement LSTM, Prophet, or other forecasting methods
3. **Risk Management**: Add portfolio optimization and risk assessment features
4. **Testing**: Expand test coverage and add backtesting capabilities
5. **Documentation**: Improve documentation and add more examples

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the existing documentation
- Review the demo scripts for usage examples

---

**Disclaimer**: This software is for educational purposes only. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

## Dependencies

The system uses the following Python libraries:

### Core Libraries
*   **pandas** - Data manipulation and analysis
*   **numpy** - Numerical computing
*   **matplotlib** - Data visualization and plotting
*   **seaborn** - Statistical data visualization
*   **statsmodels** - Time series analysis (ARIMA/SARIMA models)
*   **scikit-learn** - Machine learning metrics (RMSE, MAE)

### Additional Libraries (15-Minute System)
*   **flask** - REST API server (optional)
*   **flask-cors** - Cross-origin resource sharing for API

### Installation

**Option 1: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Manual installation**
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter flask flask-cors
```

**Option 3: Using conda**
```bash
conda install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
pip install flask flask-cors  # Flask not available in conda
```

### Python Version
- **Minimum**: Python 3.7+
- **Recommended**: Python 3.9 or 3.10 for best compatibility

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended for larger datasets)
- **Storage**: 100MB for code and sample data
- **Network**: Internet connection for installing dependencies and optional real-time data feeds
