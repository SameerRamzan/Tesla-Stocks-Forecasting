# Tesla Stocks Forecasting Using ARIMA & SARIMA

## Project Description

This repository contains a Jupyter Notebook that performs time-series analysis and forecasting on historical stock price data for Tesla, Inc. (TSLA). The primary goal is to predict future Tesla stock closing prices using ARIMA (Autoregressive Integrated Moving Average) and SARIMA (Seasonal ARIMA) models. The notebook covers data loading, preprocessing, exploratory data analysis through visualization, model fitting, forecasting, and model performance evaluation.

## Files in this Repository

*   **`Tesla Stock Dataset.csv`**: A CSV file containing daily historical stock price data for Tesla (TSLA) from June 29, 2010, to January 3, 2025.
*   **`Tesla_stock_Forecasting___ARIMA_SARIMA.ipynb`**: A Jupyter Notebook that includes the Python code for analyzing the stock data and implementing the ARIMA and SARIMA forecasting models.
*   **`README.md`**: This file, providing an overview of the project.

## Dataset

The `Tesla Stock Dataset.csv` includes the following columns:

*   **`Date`**: The trading date (YYYY-MM-DD).
*   **`Open`**: The opening stock price on the given date.
*   **`High`**: The highest stock price during the trading day.
*   **`Low`**: The lowest stock price during the trading day.
*   **`Close`**: The closing stock price for the day (this is the target variable for forecasting).
*   **`Adj Close`**: The adjusted closing price, corrected for dividends and stock splits.
*   **`Volume`**: The number of shares traded during the day.

## How to Use

1.  **Prerequisites**: Ensure you have Python installed, along with Jupyter Notebook or JupyterLab. You will also need the libraries listed in the "Dependencies" section.
2.  **Download Files**: Clone or download all files from this repository.
3.  **Open Notebook**: Launch Jupyter Notebook/Lab and open `Tesla_stock_Forecasting___ARIMA_SARIMA.ipynb`.
4.  **Run Cells**: Execute the cells in the notebook sequentially. The notebook will:
    *   Load the dataset.
    *   Preprocess and visualize the data.
    *   Perform stationarity tests.
    *   Build, fit, and evaluate ARIMA and SARIMA models.
    *   Generate and plot a 30-day forecast for Tesla's stock closing price.

## Model Information

The notebook implements two time-series forecasting models:

1.  **ARIMA (Autoregressive Integrated Moving Average)**:
    *   Parameters (p, d, q): (1, 1, 1) as used in the notebook.
    *   The model is fitted to the 'Close' price after differencing to achieve stationarity.

2.  **SARIMA (Seasonal ARIMA)**:
    *   Parameters (p, d, q): (2, 1, 2)
    *   Seasonal Parameters (P, D, Q, s): (1, 1, 1, 12)
    *   This model accounts for seasonality in the time series, with a seasonal period of 12 (likely representing months if the data were aggregated monthly, though applied to daily data here with a 365-day decomposition period for visualization).

## Results/Output

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

## Dependencies

The Jupyter Notebook uses the following Python libraries:

*   pandas
*   matplotlib
*   seaborn
*   numpy
*   statsmodels
*   scikit-learn (specifically `mean_squared_error`)
*   math

You can typically install these libraries using pip:
`pip install pandas matplotlib seaborn numpy statsmodels scikit-learn`
