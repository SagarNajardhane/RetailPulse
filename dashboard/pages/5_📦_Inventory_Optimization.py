# ============================================
# RETAILPULSE - INVENTORY OPTIMIZATION PAGE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


@st.cache_data
def load_inventory():
    inv = None
    metrics = {}
    inv_path = os.path.join(PROCESSED_DIR, "inventory_recommendations.parquet")
    metrics_path = os.path.join(PROCESSED_DIR, "inventory_metrics.json")
    if os.path.exists(inv_path):
        inv = pd.read_parquet(inv_path)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return inv, metrics


df, metrics = load_inventory()

st.title("📦 Inventory Optimization")
st.markdown(
    "Reorder points, safety stock, EOQ, and stock-status alerts. "
    "**Lead times and current stock levels use documented synthetic assumptions** "
    "because the UCI dataset has no warehouse inventory feed."
)
st.divider()

if df is None:
    st.error("⚠️ `inventory_recommendations.parquet` not found. Complete Step 10 first.")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("🔍 Filters")
if "Stock_Status" in df.columns:
    statuses = sorted(df["Stock_Status"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect("Stock status", statuses, default=statuses)
    view = df[df["Stock_Status"].isin(selected_status)] if selected_status else df
else:
    view = df

search = st.sidebar.text_input("Search product description contains")
if search and "Description" in view.columns:
    view = view[view["Description"].astype(str).str.contains(search, case=False, na=False)]

st.sidebar.markdown(f"**SKUs shown:** {len(view):,}")

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
total_skus = len(df)
critical = int((df["Stock_Status"] == "CRITICAL_STOCKOUT").sum()) if "Stock_Status" in df.columns else 0
reorder_now = int((df["Stock_Status"] == "REORDER_NOW").sum()) if "Stock_Status" in df.columns else 0
restock = metrics.get("total_restock_budget_gbp", df["Estimated_Restock_Cost"].sum() if "Estimated_Restock_Cost" in df.columns else 0)

c1.metric("SKUs Managed", f"{total_skus:,}")
c2.metric("Critical Stockouts", f"{critical:,}")
c3.metric("Reorder Now", f"{reorder_now:,}")
c4.metric("Est. Restock Budget", f"£{float(restock):,.0f}")

st.divider()

# ---------- Charts ----------
left, right = st.columns(2)

with left:
    st.markdown("### Inventory Health Mix")
    if "Stock_Status" in df.columns:
        counts = df["Stock_Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count", hole=0.35)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Current Stock vs Reorder Point")
    if {"Current_Stock", "Reorder_Point", "Stock_Status"}.issubset(df.columns):
        sample = df.sample(min(200, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x="Reorder_Point",
            y="Current_Stock",
            color="Stock_Status",
            hover_data=[c for c in ["StockCode", "Description", "Safety_Stock", "EOQ"] if c in sample.columns],
            opacity=0.75,
        )
        max_v = max(sample["Reorder_Point"].max(), sample["Current_Stock"].max())
        fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                                 name="Stock = ROP", line=dict(dash="dash", color="red")))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Urgent tables ----------
st.markdown("### 🚨 Urgent Actions")
urgent_statuses = ["CRITICAL_STOCKOUT", "REORDER_NOW"]
if "Stock_Status" in df.columns:
    urgent = df[df["Stock_Status"].isin(urgent_statuses)].copy()
    if "Estimated_Restock_Cost" in urgent.columns:
        urgent = urgent.sort_values("Estimated_Restock_Cost", ascending=False)
    cols = [c for c in [
        "StockCode", "Description", "Stock_Status", "Current_Stock", "Reorder_Point",
        "Safety_Stock", "Lead_Time_Days", "EOQ", "Recommended_Order_Qty", "Estimated_Restock_Cost"
    ] if c in urgent.columns]
    st.dataframe(urgent[cols].head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download urgent reorder list (CSV)",
        data=urgent[cols].to_csv(index=False),
        file_name="urgent_inventory_actions.csv",
        mime="text/csv",
    )
else:
    st.info("Stock_Status column missing.")

st.divider()
st.markdown("### Filtered Inventory Table")
show_cols = [c for c in [
    "StockCode", "Description", "Avg_Daily_Demand", "Std_Daily_Demand", "Lead_Time_Days",
    "Safety_Stock", "Reorder_Point", "EOQ", "Current_Stock", "Stock_Status",
    "Recommended_Order_Qty", "Estimated_Restock_Cost"
] if c in view.columns]
st.dataframe(view[show_cols].head(300), use_container_width=True, hide_index=True)

st.divider()
st.markdown("### Inventory Metrics (JSON)")
st.json(metrics if metrics else {"message": "inventory_metrics.json not found"})