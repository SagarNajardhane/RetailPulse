# ============================================
# RETAILPULSE - DEMAND FORECASTING PAGE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecasting", page_icon="📉", layout="wide")

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


@st.cache_data
def load_forecast_data():
    out = {}
    # Prefer ensemble results; fallback to Step 7 test results
    ens_path = os.path.join(PROCESSED_DIR, "ensemble_test_results.csv")
    step7_path = os.path.join(PROCESSED_DIR, "prophet_test_results.csv")
    ts_path = os.path.join(PROCESSED_DIR, "daily_timeseries_features.parquet")
    metrics_path = os.path.join(PROCESSED_DIR, "forecasting_metrics.json")

    if os.path.exists(ens_path):
        df = pd.read_csv(ens_path)
        df["Date"] = pd.to_datetime(df["Date"])
        out["test"] = df
        out["mode"] = "ensemble"
    elif os.path.exists(step7_path):
        df = pd.read_csv(step7_path)
        df["Date"] = pd.to_datetime(df["Date"])
        # Normalize column names
        rename = {}
        if "Actual" not in df.columns and "y" in df.columns:
            rename["y"] = "Actual"
        if "Predicted" in df.columns:
            rename["Predicted"] = "Ridge_Predicted"
        df = df.rename(columns=rename)
        out["test"] = df
        out["mode"] = "ridge"
    else:
        out["test"] = None
        out["mode"] = None

    if os.path.exists(ts_path):
        ts = pd.read_parquet(ts_path)
        ts["Date"] = pd.to_datetime(ts["Date"])
        out["history"] = ts
    else:
        out["history"] = None

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            out["metrics"] = json.load(f)
    else:
        out["metrics"] = {}

    return out


data = load_forecast_data()

st.title("📉 Demand Forecasting")
st.markdown("Daily revenue forecast from Ridge/Prophet-style model + LSTM ensemble (where available).")
st.divider()

if data["test"] is None:
    st.error("⚠️ Forecast test results not found. Complete Steps 7–8 and ensure CSV files exist in `data/processed/`.")
    st.stop()

test = data["test"].copy()
metrics = data["metrics"]
mode = data["mode"]

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)

actual_sum = test["Actual"].sum() if "Actual" in test.columns else 0
pred_col = None
for c in ["Ensemble_Predicted", "Ridge_Predicted", "Predicted", "LSTM_Predicted"]:
    if c in test.columns:
        pred_col = c
        break

pred_sum = test[pred_col].sum() if pred_col else 0
mape = metrics.get("ensemble_mape") or metrics.get("daily_mape") or metrics.get("ridge_mape")
winner = metrics.get("winner_model", mode or "N/A")

col1.metric("Test Actual Revenue", f"£{actual_sum:,.0f}")
col2.metric("Test Predicted Revenue", f"£{pred_sum:,.0f}")
col3.metric("Best MAPE", f"{mape}%" if mape is not None else "N/A")
col4.metric("Winning Model", str(winner))

st.divider()

# ---------- Main forecast chart ----------
st.markdown("### Actual vs Predicted (Test Period)")

fig = go.Figure()
if "Actual" in test.columns:
    fig.add_trace(go.Scatter(x=test["Date"], y=test["Actual"], name="Actual",
                             line=dict(color="black", width=2.5)))
if "Ridge_Predicted" in test.columns:
    fig.add_trace(go.Scatter(x=test["Date"], y=test["Ridge_Predicted"], name="Ridge / Step 7",
                             line=dict(color="orange", width=2, dash="dash")))
if "LSTM_Predicted" in test.columns:
    fig.add_trace(go.Scatter(x=test["Date"], y=test["LSTM_Predicted"], name="LSTM",
                             line=dict(color="purple", width=2, dash="dot")))
if "Ensemble_Predicted" in test.columns:
    fig.add_trace(go.Scatter(x=test["Date"], y=test["Ensemble_Predicted"], name="Ensemble",
                             line=dict(color="green", width=3)))
elif "Predicted" in test.columns:
    fig.add_trace(go.Scatter(x=test["Date"], y=test["Predicted"], name="Predicted",
                             line=dict(color="green", width=3)))

fig.update_layout(height=450, yaxis_title="Revenue (£)", xaxis_title="Date",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
fig.update_yaxes(tickformat="£,.0f")
st.plotly_chart(fig, use_container_width=True)

# ---------- History context ----------
if data["history"] is not None:
    st.markdown("### Historical Daily Revenue Context")
    hist = data["history"].tail(180)
    fig_h = px.line(hist, x="Date", y="Daily_Revenue", title="Last ~180 days of history")
    fig_h.update_layout(height=320, yaxis_tickformat="£,.0f")
    st.plotly_chart(fig_h, use_container_width=True)

# ---------- Metrics table ----------
st.markdown("### Model Metrics")
if metrics:
    st.json(metrics)
else:
    st.info("No `forecasting_metrics.json` found — metrics may only exist in notebooks.")

# ---------- Data table ----------
st.markdown("### Forecast Test Data")
st.dataframe(test.head(200), use_container_width=True, hide_index=True)

st.caption("Note: Daily MAPE is often higher than monthly MAPE because single days are noisy.")