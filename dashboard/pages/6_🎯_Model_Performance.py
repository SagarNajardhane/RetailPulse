# ============================================
# RETAILPULSE - MODEL PERFORMANCE PAGE
# ============================================

import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px

st.set_page_config(page_title="Model Performance", page_icon="🎯", layout="wide")

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


@st.cache_data
def load_all_metrics():
    files = {
        "segmentation": "segmentation_metrics.json",
        "forecasting": "forecasting_metrics.json",
        "churn": "churn_metrics.json",
        "inventory": "inventory_metrics.json",
    }
    out = {}
    for key, name in files.items():
        path = os.path.join(PROCESSED_DIR, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                out[key] = json.load(f)
        else:
            out[key] = None
    return out


metrics = load_all_metrics()

st.title("🎯 Model Performance Center")
st.markdown("Single place to review all ML / analytics KPIs used in RetailPulse.")
st.divider()

# ---------- Scorecards ----------
s1, s2, s3, s4 = st.columns(4)

seg = metrics.get("segmentation") or {}
fore = metrics.get("forecasting") or {}
churn = metrics.get("churn") or {}
inv = metrics.get("inventory") or {}

sil = seg.get("silhouette_score", "N/A")
mape = fore.get("ensemble_mape", fore.get("daily_mape", fore.get("ridge_mape", "N/A")))
auc = churn.get("roc_auc", "N/A")
critical = inv.get("critical_stockout_skus", "N/A")

s1.metric("Segmentation Silhouette", f"{sil}", help="Target ≥ 0.45 (real-world may be lower)")
s2.metric("Forecast MAPE", f"{mape}", help="Daily MAPE often higher than monthly")
s3.metric("Churn ROC-AUC", f"{auc}", help="Target ≥ 0.88")
s4.metric("Critical Inventory SKUs", f"{critical}")

st.divider()

# ---------- Target checklist ----------
st.markdown("### Project Target Checklist (Honest)")

rows = []
rows.append({
    "Area": "Customer Segmentation",
    "Metric": "Silhouette",
    "Target": "≥ 0.45",
    "Actual": seg.get("silhouette_score"),
    "Met?": bool(seg.get("target_achieved", False)) if seg else False,
})
rows.append({
    "Area": "Demand Forecasting",
    "Metric": "MAPE",
    "Target": "≤ 12%",
    "Actual": mape,
    "Met?": bool(fore.get("target_achieved", fore.get("target_achieved_daily", False))) if fore else False,
})
rows.append({
    "Area": "Churn Prediction",
    "Metric": "ROC-AUC",
    "Target": "≥ 0.88",
    "Actual": auc,
    "Met?": bool(churn.get("target_achieved", False)) if churn else False,
})

check_df = pd.DataFrame(rows)
st.dataframe(check_df, use_container_width=True, hide_index=True)

st.caption("We report actual metrics. Missing a stretch target on real retail data is acceptable if methods and limitations are documented.")

st.divider()

# ---------- Detail panels ----------
t1, t2, t3, t4 = st.tabs(["Segmentation", "Forecasting", "Churn", "Inventory"])

with t1:
    if seg:
        st.json(seg)
    else:
        st.warning("segmentation_metrics.json missing")

with t2:
    if fore:
        st.json(fore)
        # Simple bar if multiple MAPEs exist
        plot_data = []
        for k in ["ridge_mape", "lstm_mape", "ensemble_mape", "daily_mape", "monthly_mape"]:
            if k in fore and fore[k] is not None:
                plot_data.append({"Metric": k, "Value": fore[k]})
        if plot_data:
            fig = px.bar(pd.DataFrame(plot_data), x="Metric", y="Value", title="Forecast error metrics")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("forecasting_metrics.json missing")

with t3:
    if churn:
        st.json(churn)
        plot_data = []
        for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if k in churn and churn[k] is not None:
                plot_data.append({"Metric": k, "Value": churn[k]})
        if plot_data:
            fig = px.bar(pd.DataFrame(plot_data), x="Metric", y="Value", title="Churn model metrics")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("churn_metrics.json missing")

with t4:
    if inv:
        st.json(inv)
        if "status_breakdown" in inv and isinstance(inv["status_breakdown"], dict):
            sb = pd.DataFrame([
                {"Status": k, "Count": v} for k, v in inv["status_breakdown"].items()
            ])
            fig = px.bar(sb, x="Status", y="Count", title="Inventory status breakdown")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("inventory_metrics.json missing")

st.divider()
st.markdown(
    """
    ### How to interpret this page
    - **Segmentation**: cluster quality (silhouette) + business labels  
    - **Forecasting**: MAE / RMSE / MAPE and ensemble winner  
    - **Churn**: classification quality + risk tier counts  
    - **Inventory**: operational alerts (with synthetic stock/lead-time assumptions clearly marked)
    """
)