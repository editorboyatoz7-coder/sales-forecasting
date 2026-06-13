"""
Streamlit dashboard for Sales Forecasting project.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from src.preprocessing import load_data, build_features, train_test_split_by_date
from src.evaluation import evaluate
from src.forecast_future import forecast_future

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📈 Sales Forecasting Dashboard")
st.caption("Internship Project — Sales Forecasting System")

DATA_PATH = "data/sales_data.csv"
MODEL_PATH = "outputs/best_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run `python src/generate_data.py` first.")
    st.stop()

df = load_data(DATA_PATH)

# Sidebar
st.sidebar.header("Settings")
n_days_history = st.sidebar.slider("Days of history to show", 30, len(df), 180)
n_days_forecast = st.sidebar.slider("Days to forecast", 7, 90, 30)

# Historical sales chart
st.subheader("Historical Sales")
recent = df.tail(n_days_history)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(recent["date"], recent["sales"], label="Sales", color="#1f77b4")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

# Summary stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
col2.metric("Avg Daily Sales", f"{df['sales'].mean():.1f}")
col3.metric("Max Sales", int(df["sales"].max()))
col4.metric("Min Sales", int(df["sales"].min()))

st.divider()

# Model comparison
st.subheader("Model Comparison")
comparison_path = "outputs/model_comparison.csv"
if os.path.exists(comparison_path):
    comparison_df = pd.read_csv(comparison_path)
    st.dataframe(comparison_df, use_container_width=True)
else:
    st.warning("Model comparison results not found. Run `python -m src.train_models` first.")

# Test predictions vs actual
st.subheader("Test Set: Actual vs Predicted")
preds_path = "outputs/test_predictions.csv"
if os.path.exists(preds_path):
    preds_df = pd.read_csv(preds_path, parse_dates=["date"])
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(preds_df["date"], preds_df["sales"], label="Actual", color="#1f77b4")
    ax2.plot(preds_df["date"], preds_df["prediction"], label="Predicted", color="#ff7f0e", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("Test predictions not found. Run `python -m src.train_models` first.")

st.divider()

# Future forecast
st.subheader(f"Future Forecast — Next {n_days_forecast} Days")
if os.path.exists(MODEL_PATH):
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            forecast_df = forecast_future(n_days=n_days_forecast)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(df["date"].tail(60), df["sales"].tail(60), label="Recent Actual", color="#1f77b4")
        ax3.plot(forecast_df["date"], forecast_df["predicted_sales"], label="Forecast", color="#2ca02c", linestyle="--")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Sales")
        ax3.legend()
        st.pyplot(fig3)
        st.dataframe(forecast_df, use_container_width=True)
else:
    st.warning("Trained model not found. Run `python -m src.train_models` first.")

st.divider()
st.caption("Built with Streamlit | Sales Forecasting Internship Project")
