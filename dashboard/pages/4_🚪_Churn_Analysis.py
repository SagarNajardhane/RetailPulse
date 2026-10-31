# ============================================
# RETAILPULSE - CHURN ANALYSIS PAGE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Analysis", page_icon="🚪", layout="wide")

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


@st.cache_data
def load_churn_data():
    scored = None
    metrics = {}
    scored_path = os.path.join(PROCESSED_DIR, "customer_churn_scored.parquet")
    metrics_path = os.path.join(PROCESSED_DIR, "churn_metrics.json")

    if os.path.exists(scored_path):
        scored = pd.read_parquet(scored_path)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return scored, metrics


df, metrics = load_churn_data()

st.title("🚪 Customer Churn Analysis")
st.markdown("XGBoost risk scores, risk tiers, and actionable at-risk customer lists.")
st.divider()

if df is None:
    st.error("⚠️ `customer_churn_scored.parquet` not found. Complete Step 9 first.")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("🔍 Filters")
if "Churn_Risk_Tier" in df.columns:
    tiers = sorted(df["Churn_Risk_Tier"].dropna().unique().tolist())
    selected = st.sidebar.multiselect("Risk tiers", tiers, default=tiers)
    view = df[df["Churn_Risk_Tier"].isin(selected)] if selected else df
else:
    view = df
    selected = []

min_prob = 0.0
if "Churn_Probability" in df.columns:
    min_prob = st.sidebar.slider("Min churn probability", 0.0, 1.0, 0.0, 0.05)
    view = view[view["Churn_Probability"] >= min_prob]

st.sidebar.markdown(f"**Customers shown:** {len(view):,}")

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
churn_rate = df["Is_Churned"].mean() * 100 if "Is_Churned" in df.columns else 0
avg_prob = df["Churn_Probability"].mean() * 100 if "Churn_Probability" in df.columns else 0
high = (df["Churn_Risk_Tier"].astype(str).str.contains("High", case=False)).sum() if "Churn_Risk_Tier" in df.columns else 0
auc = metrics.get("roc_auc", "N/A")

c1.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
c2.metric("Avg Churn Probability", f"{avg_prob:.1f}%")
c3.metric("High-Risk Customers", f"{high:,}")
c4.metric("XGBoost ROC-AUC", f"{auc}")

if metrics:
    target_ok = metrics.get("target_achieved", False)
    if target_ok:
        st.success(f"Target AUC ≥ 0.88: ACHIEVED (AUC = {auc})")
    else:
        st.warning(f"Target AUC ≥ 0.88: not met (AUC = {auc}). Reported honestly.")

st.divider()

# ---------- Charts ----------
left, right = st.columns(2)

with left:
    st.markdown("### Risk Tier Distribution")
    if "Churn_Risk_Tier" in df.columns:
        tier_counts = df["Churn_Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig = px.pie(tier_counts, names="Tier", values="Count", hole=0.4)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Churn_Risk_Tier column found.")

with right:
    st.markdown("### Churn Probability Distribution")
    if "Churn_Probability" in df.columns:
        fig = px.histogram(df, x="Churn_Probability", nbins=40, color_discrete_sequence=["#e74c3c"])
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Churn_Probability column found.")

st.divider()

# ---------- Scatter ----------
st.markdown("### Behavior vs Risk")
x_col = "Frequency" if "Frequency" in df.columns else None
y_col = "Monetary" if "Monetary" in df.columns else None
color_col = "Churn_Risk_Tier" if "Churn_Risk_Tier" in df.columns else ("Is_Churned" if "Is_Churned" in df.columns else None)

if x_col and y_col:
    fig = px.scatter(
        view,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=[c for c in ["Customer ID", "Churn_Probability", "Recency"] if c in view.columns],
        opacity=0.65,
    )
    fig.update_layout(height=480, yaxis_title="Monetary (£)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- High-risk table ----------
st.markdown("### 🚨 Priority Intervention List (High Risk)")
if "Churn_Risk_Tier" in df.columns and "Churn_Probability" in df.columns:
    high_df = df[df["Churn_Risk_Tier"].astype(str).str.contains("High", case=False)].copy()
    high_df = high_df.sort_values("Churn_Probability", ascending=False)
    cols = [c for c in [
        "Customer ID", "Churn_Probability", "Churn_Risk_Tier",
        "Recency", "Frequency", "Monetary", "Total_Orders", "Cluster_Name"
    ] if c in high_df.columns]
    st.dataframe(high_df[cols].head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download high-risk customers (CSV)",
        data=high_df[cols].to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )
else:
    st.info("High-risk list needs Churn_Risk_Tier and Churn_Probability.")

st.divider()
st.markdown("### Model Metrics (JSON)")
st.json(metrics if metrics else {"message": "churn_metrics.json not found"})