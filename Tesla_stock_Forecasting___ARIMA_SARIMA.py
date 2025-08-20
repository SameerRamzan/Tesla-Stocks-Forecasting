#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller #(Augmented Dickey-Fuller test)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # autocorrelation and partial autocorrelation plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")


# # Load & Inspect the Dataset

# In[ ]:


tesla_df = pd.read_csv('/content/Tesla Stock Dataset.csv')
tesla_df.head()


# In[ ]:


print(tesla_df.describe())
print(tesla_df.info())


# In[ ]:


# Convert the Date column to a datetime format and set it as the index.
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])
tesla_df.set_index('Date', inplace=True)


# In[ ]:


#Select the close column for analysis
close_data = tesla_df['Close']
close_data.dropna()


# In[ ]:


#Plot the time series
plt.figure(figsize=(12,6))
plt.plot(close_data, label='Tesla Stock Closing Prices')
plt.title('Tesla Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# # Decompose The Time Series

# In[ ]:


#decompose the time series to observe the trends, seasonality and residuals
decomposition = seasonal_decompose(close_data, model='additive', period=365)
#plot the decomposition
plt.rcParams['figure.figsize'] = (20, 10)  # Adjust width and height as desired
decomposition.plot()
plt.show()


# 1. The `additive model` assumes that the time series is a sum of its components (trend + seasonality + residual). This is suitable when the seasonal fluctuations are relatively constant over time. Alternatively, a `multiplicative model` can be used if the seasonal variations change proportionally with the level of the time series.
# 2. `period=365` This crucial parameter defines the length of the seasonal cycle. In this case, it's set to 365, implying that the code is looking for patterns that repeat every 365 periods (likely representing a 365-day cycle, which could roughly correspond to Yearly seasonality in daily data).

# 1. **Trend**: The seasonal_decompose function isolates the long-term progression of the data, which is referred to as the trend. It essentially smooths out short-term fluctuations to reveal the underlying direction of the series.
# 2. **Seasonality**: The function identifies patterns that repeat at regular intervals (defined by the period parameter), which is the seasonality. This could be daily, weekly, monthly, or any other recurring pattern in the data.
# 3. **Residual**: After extracting the trend and seasonality, whatever variation remains is categorized as the residual. These are the unpredictable fluctuations or noise in the time series.

# # Test For Stationarity

# In[ ]:


#perform Augmented Dicky Fuller Test
result = adfuller(close_data)
print("ADF Statistic", result[0])
print("Print P value", result[1])
print("Critical Values", result[4])

#interpret the results
if result[1] <= 0.05:
  print("Time Series Data is Stationary")
else:
  print("Time Series Data is Non-Stationary, Differencing is required")


# 1. **Low p-value** (≤ 0.05) and a significantly negative ADF Statistic: The time series is likely stationary.
# 2. **High p-value** (> 0.05) and an ADF Statistic not much lower than Critical Values: The time series is likely non-stationary. You might need to apply differencing to make it stationary before using certain time series models.

# ## Differencing The Data

# In[ ]:


#Apply first-order differencing
data_diff = close_data.diff()

#plot the differenced data
plt.plot(data_diff, label='First Order Differenced Data')
plt.title('Differenced Tesla Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.legend()
plt.show()


# **Differencing** is a time series preprocessing technique used to make a time series stationary.
# 
# - How Much Differencing is Enough?
# 
# Start with **first-order differencing** (subtracting
# 𝑦
# 𝑡
# −
# 𝑦
# 𝑡
# −
# 1
# y
# t
# ​
#  −y
# t−1
# ​
#  ).
# If the series is still non-stationary, apply **second-order differencing** (difference the differenced series:
# 𝑦
# 𝑡
# ′
# −
# 𝑦
# 𝑡
# −
# 1
# ′
# y
# t
# ′
# ​
#  −y
# t−1
# ′
# ​
#  ).
# 
# Avoid over-differencing, as it can lead to loss of information and increased noise.

# In[ ]:


#apply adfuller test to check the stationary of differenced data
diff = adfuller(data_diff.dropna())
print("ADF Statistic", diff[0])
print("Print P value", diff[1])
print("Critical Values", diff[4])

#interpret the results
if diff[1] <= 0.05:
  print("Data is Stationary.")
else:
  print("Data is Non-Stationary, Differencing is required")


# Finally, We have Stationary dataset. Now we can move on to next step, which is
# `ACf` and `PACF` plots.
# 
# 

# # ACF and PACF Plots

# In[ ]:


#plot ACF and PACF on differenced data
fig, ax = plt.subplots(1, 2, figsize=(12,6))
plot_acf(data_diff, ax=ax[0], lags=30, title='ACF')
plot_pacf(data_diff, ax=ax[1], lags=30, title='PACF')
plt.show()


# - **Data is already stationary**: If your time series is already stationary (as indicated by our ADF test), there might not be any significant autocorrelations left to detect. In such cases, ACF and PACF plots could be flat.

# In[ ]:


#plot ACF and PACF on original data
fig, ax = plt.subplots(1, 2, figsize=(12,6))
plot_acf(close_data, ax=ax[0], lags=30, title='ACF')
plot_pacf(close_data, ax=ax[1], lags=30, title='PACF')
plt.show()


# - It suggests your original data had autocorrelations, but they were removed by differencing.
# - **Model diagnostics**: Even if no clear spikes are present, you can try fitting an ARIMA model with small values of p and q (e.g., ARIMA(1, d, 1)) and then check the model diagnostics. If the residuals are white noise, it suggests the model adequately captures the time series dynamics.

# # Build The ARIMA Model

# **Understanding ARIMA Parameters**
# 
# 1. **p (AR order)**: Represents the number of autoregressive terms in the model. It indicates how many past values of the time series are used to predict the current value.
# 2. **d (Differencing order)**: Represents the number of times the time series needs to be differenced to make it stationary. It addresses trends and seasonality.
# 3. **q (MA order)**: Represents the number of moving average terms in the model. It indicates how many past forecast errors are used to predict the current value.

# In[ ]:


# define parameters
p, d, q = 1, 1, 1
#define the model
arima_model = ARIMA(close_data, order=(p, d, q))
arima_result = arima_model.fit()

#print the summary of the ARIMA Model
print(arima_result.summary())


# In[ ]:


residuals = arima_result.resid

fig, ax = plt.subplots(1, 2, figsize=(12,6))
plot_acf(residuals, ax=ax[0], lags=30, title='ACF of Residuals')
plot_pacf(residuals, ax=ax[1], lags=30, title='PACF of Residuals')
plt.show()


# In[ ]:


#plot the fitted values against the actual data
plt.plot(close_data, label='Actual Data')
plt.plot(arima_result.fittedvalues, color='red', label='Fitted Values')
plt.title('ARIMA Model Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# # Build The Sarima Model

# In[ ]:


#Define the model
p, d, q = 2, 1, 2
P, D, Q, seasonal_period =1, 1, 1, 12
sarima_model = SARIMAX(close_data, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
sarima_result = sarima_model.fit()

#summary of the model
print(sarima_result.summary())


# In[ ]:


#plot the fitted values against the actual values
plt.plot(close_data, label='Actual Data')
plt.plot(sarima_result.fittedvalues, color='green', label='Fitted')
plt.title('SARIMA Model - Actual vs Fitted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# # Forecast Future Prices
# 

# In[ ]:


#forecast the next 30 days
forecast_steps = 30
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(close_data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_df = pd.DataFrame({
    'Forecast': forecast.predicted_mean,
    'Lower Bound': forecast.conf_int()['lower Close'],
    'Upper Bound': forecast.conf_int()['upper Close']
})


# In[ ]:


#plot the forecast
plt.plot(close_data, label='Actual Data')
plt.plot(forecast_df['Forecast'], label='Forecast', color='orange')
plt.fill_between(forecast_df.index, forecast_df['Lower Bound'], forecast_df['Upper Bound'])
plt.title('SARIMA Forecast for Tesla Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:


forecast_df.head()


# #Evaluate The Model Performance

# In[ ]:


# Calculate RMSE for ARIMA and SARIMA
arima_predictions = arima_result.predict(start=close_data.index[0], end=close_data.index[-1])
sarima_predictions = sarima_result.fittedvalues

arima_rmse = sqrt(mean_squared_error(close_data, arima_predictions))
sarima_rmse = sqrt(mean_squared_error(close_data, sarima_predictions))

print(f"ARIMA RMSE: {arima_rmse}")
print(f"SARIMA RMSE: {sarima_rmse}")

