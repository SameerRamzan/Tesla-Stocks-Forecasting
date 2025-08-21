"""Tesla Stock Forecasting Dashboard - Streamlit Frontend."""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Tesla Stock Forecasting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL
BACKEND_URL = "http://localhost:8000/api/v1"

# Title and header
st.title("🚗 Tesla Stock Forecasting & Analytics")
st.markdown("Production-grade forecasting system for TSLA stock analysis")

# Sidebar configuration
st.sidebar.header("Configuration")
interval = st.sidebar.selectbox(
    "Data Interval",
    ["1d", "1h", "5m"],
    index=0
)

# Main navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📈 Forecasting", 
        "🔬 Model Comparison",
        "🧪 What-If Analysis",
        "📋 Data Quality",
        "⚙️ Data Management"
    ]
)

@st.cache_data(ttl=300)
def fetch_data(endpoint, params=None):
    """Fetch data from backend API with caching."""
    try:
        response = requests.get(f"{BACKEND_URL}/{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def post_data(endpoint, data):
    """Post data to backend API."""
    try:
        response = requests.post(f"{BACKEND_URL}/{endpoint}", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

if page == "📊 Dashboard":
    st.header("Market Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Interval", interval)
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        health_data = fetch_data("health")
        if health_data:
            status = health_data.get("status", "unknown")
            st.metric("API Status", status.upper(), delta="Healthy" if status == "ok" else "Error")
    
    # Price chart
    st.subheader("Tesla Stock Price History")
    
    price_data = fetch_data("data/prices", {"interval": interval, "limit": 200})
    
    if price_data and price_data.get("data"):
        df = pd.DataFrame(price_data["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="TSLA"
        ))
        
        fig.update_layout(
            title="Tesla Stock Price (OHLC)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        vol_fig = px.bar(df, x="timestamp", y="volume", title="Trading Volume")
        vol_fig.update_layout(height=300)
        st.plotly_chart(vol_fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Latest Close", f"${df['close'].iloc[-1]:.2f}")
        with col2:
            daily_return = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
            st.metric("Daily Return", f"{daily_return:.2f}%")
        with col3:
            st.metric("52W High", f"${df['high'].max():.2f}")
        with col4:
            st.metric("52W Low", f"${df['low'].min():.2f}")
    
    else:
        st.warning("No price data available. Please check data ingestion.")

elif page == "📈 Forecasting":
    st.header("Stock Price Forecasting")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Select Model",
            [
                "sarima_212_111_12",
                "arima_111", 
                "xgboost",
                "naive",
                "sma_5",
                "sma_20",
                "ewma_0.3"
            ]
        )
    
    with col2:
        horizons = st.multiselect(
            "Forecast Horizons",
            [1, 2, 3, 5, 10, 20],
            default=[1, 5, 10]
        )
    
    if st.button("🔮 Generate Forecast"):
        if not horizons:
            st.error("Please select at least one forecast horizon")
        else:
            with st.spinner("Generating forecast..."):
                forecast_request = {
                    "symbol": "TSLA",
                    "interval": interval,
                    "horizons": horizons,
                    "model_name": model_name,
                    "include_pi": True
                }
                
                forecast_data = post_data("forecast", forecast_request)
                
                if forecast_data:
                    st.success(f"Forecast generated using {forecast_data['model_name']}")
                    
                    # Display forecast results
                    forecasts = forecast_data["forecasts"]
                    
                    if forecasts:
                        df_forecast = pd.DataFrame(forecasts)
                        
                        # Forecast table
                        st.subheader("Forecast Results")
                        
                        display_df = df_forecast.copy()
                        display_df["target_ts"] = pd.to_datetime(display_df["target_ts"])
                        display_df["yhat"] = display_df["yhat"].round(2)
                        
                        if "yhat_lower" in display_df.columns and display_df["yhat_lower"].notna().any():
                            display_df["yhat_lower"] = display_df["yhat_lower"].round(2)
                            display_df["yhat_upper"] = display_df["yhat_upper"].round(2)
                            display_df["confidence_width"] = (display_df["yhat_upper"] - display_df["yhat_lower"]).round(2)
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Forecast visualization
                        fig = go.Figure()
                        
                        # Add forecast points
                        fig.add_trace(go.Scatter(
                            x=display_df["horizon"],
                            y=display_df["yhat"],
                            mode='markers+lines',
                            name='Forecast',
                            line=dict(color='blue', width=2),
                            marker=dict(size=8)
                        ))
                        
                        # Add confidence intervals if available
                        if "yhat_lower" in display_df.columns and display_df["yhat_lower"].notna().any():
                            fig.add_trace(go.Scatter(
                                x=display_df["horizon"],
                                y=display_df["yhat_upper"],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,100,80,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=display_df["horizon"],
                                y=display_df["yhat_lower"],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,100,80,0)',
                                name='95% Confidence Interval',
                                fillcolor='rgba(0,100,80,0.2)'
                            ))
                        
                        fig.update_layout(
                            title=f"Tesla Stock Price Forecast - {model_name}",
                            xaxis_title="Forecast Horizon (days)",
                            yaxis_title="Predicted Price ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.warning("No forecast data returned")

elif page == "🔬 Model Comparison":
    st.header("Model Backtesting & Comparison")
    
    # Model selection for backtesting
    available_models = [
        "naive", "sma_5", "sma_20", "ewma_0.3", 
        "arima_111", "sarima_212_111_12", "xgboost"
    ]
    
    selected_models = st.multiselect(
        "Select Models to Compare",
        available_models,
        default=["naive", "arima_111", "sarima_212_111_12"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        validation_type = st.selectbox(
            "Validation Type",
            ["walk_forward", "expanding_window"]
        )
    
    with col2:
        max_iterations = st.slider("Max Iterations", 10, 100, 30)
    
    if st.button("🧪 Run Backtest") and selected_models:
        with st.spinner("Running backtest..."):
            backtest_request = {
                "model_names": selected_models,
                "interval": interval,
                "validation_type": validation_type,
                "max_iterations": max_iterations,
                "initial_train_size": 50,
                "test_size": 1
            }
            
            backtest_data = post_data("forecast/backtest", backtest_request)
            
            if backtest_data and backtest_data.get("results"):
                st.success(f"Backtest completed! Champion model: {backtest_data.get('champion_model', 'N/A')}")
                
                # Results table
                results_df = pd.DataFrame([
                    {
                        "Model": result["model_name"],
                        "Status": result["status"],
                        "RMSE": round(result["metrics"]["rmse"], 4),
                        "MAE": round(result["metrics"]["mae"], 4),
                        "MAPE": round(result["metrics"]["mape"], 2),
                        "R²": round(result["metrics"]["r2"], 4),
                        "Directional Accuracy": round(result["metrics"]["directional_accuracy"], 2),
                        "Predictions": result["total_predictions"]
                    }
                    for result in backtest_data["results"]
                ])
                
                st.subheader("Backtest Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Performance visualization
                metrics_to_plot = ["RMSE", "MAE", "MAPE", "R²"]
                
                fig_metrics = go.Figure()
                
                for metric in metrics_to_plot:
                    if metric in results_df.columns:
                        fig_metrics.add_trace(go.Bar(
                            name=metric,
                            x=results_df["Model"],
                            y=results_df[metric]
                        ))
                
                fig_metrics.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)

elif page == "📋 Data Quality":
    st.header("Data Quality Report")
    
    if st.button("🔍 Run Data Quality Check"):
        with st.spinner("Checking data quality..."):
            quality_data = fetch_data(f"ingestion/data-quality/{interval}")
            
            if quality_data:
                st.success("Data quality check completed")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", quality_data.get("record_count", 0))
                
                with col2:
                    date_range = quality_data.get("date_range", {})
                    start_date = date_range.get("start", "N/A")
                    st.metric("Data Start", start_date[:10] if start_date != "N/A" else "N/A")
                
                with col3:
                    end_date = date_range.get("end", "N/A")
                    st.metric("Data End", end_date[:10] if end_date != "N/A" else "N/A")
                
                # Quality checks
                st.subheader("Data Validation Results")
                
                price_consistency = quality_data.get("price_consistency", {})
                
                checks = [
                    ("Low > High violations", price_consistency.get("low_high_violations", 0)),
                    ("Low > Close violations", price_consistency.get("low_close_violations", 0)),
                    ("High < Close violations", price_consistency.get("high_close_violations", 0)),
                    ("Negative volume records", quality_data.get("negative_volume", 0)),
                    ("Zero volume records", quality_data.get("zero_volume", 0))
                ]
                
                for check_name, value in checks:
                    color = "green" if value == 0 else "red"
                    status = "✅ PASS" if value == 0 else "❌ FAIL"
                    st.markdown(f"**{check_name}**: {value} ({status})")

elif page == "⚙️ Data Management":
    st.header("Data Management")
    
    # Data ingestion
    st.subheader("Data Ingestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", value=datetime.now())
    
    with col2:
        lookback_days = st.slider("Lookback Days (if no start date)", 1, 365, 30)
    
    if st.button("📥 Ingest Data"):
        with st.spinner("Ingesting data..."):
            ingest_request = {
                "interval": interval,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "lookback_days": lookback_days
            }
            
            ingest_data = post_data("ingestion/ingest", ingest_request)
            
            if ingest_data:
                if ingest_data.get("status") == "success":
                    st.success(f"✅ Data ingestion completed!")
                    st.json(ingest_data)
                else:
                    st.error(f"❌ Data ingestion failed: {ingest_data.get('message', 'Unknown error')}")
    
    # Feature computation
    st.subheader("Feature Engineering")
    
    if st.button("🔧 Compute Features"):
        with st.spinner("Computing features..."):
            feature_request = {
                "interval": interval,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "lookback_days": 100
            }
            
            feature_data = post_data("features/compute", feature_request)
            
            if feature_data:
                if feature_data.get("status") == "success":
                    st.success(f"✅ Feature computation completed!")
                    st.json(feature_data)
                else:
                    st.error(f"❌ Feature computation failed: {feature_data.get('message', 'Unknown error')}")

# Footer
st.markdown("---")
st.markdown("**Tesla Stock Forecasting System** | Built with FastAPI + Streamlit | Version 0.1.0")