# ============================================
# RETAILPULSE - CUSTOMER SEGMENTATION PAGE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

# Paths
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Load data
@st.cache_data
def load_segmentation_data():
    # Try segmented data first, then customer features
    for filename in ["customers_segmented.parquet", "customer_features.parquet"]:
        path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(path):
            return pd.read_parquet(path)
    return None

@st.cache_data
def load_metrics():
    path = os.path.join(PROCESSED_DIR, "segmentation_metrics.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

df = load_segmentation_data()
metrics = load_metrics()

st.title("👥 Customer Segmentation Explorer")
st.markdown("AI-driven customer clusters based on RFM analysis and K-Means clustering.")
st.divider()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("🔍 Segment Filters")
    
    has_clusters = 'Cluster_Name' in df.columns or 'Cluster' in df.columns
    
    if has_clusters:
        if 'Cluster_Name' in df.columns:
            segments = sorted(df['Cluster_Name'].unique())
            selected_segment = st.sidebar.selectbox("Select Segment", ["All"] + segments)
            if selected_segment != "All":
                df_filtered = df[df['Cluster_Name'] == selected_segment]
            else:
                df_filtered = df
        else:
            df_filtered = df
        
        st.sidebar.markdown(f"**Customers Shown:** {len(df_filtered):,}")
    else:
        df_filtered = df
    
    # --- KPI CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    avg_monetary = df['Monetary'].mean() if 'Monetary' in df.columns else 0
    avg_frequency = df['Frequency'].mean() if 'Frequency' in df.columns else 0
    avg_recency = df['Recency'].mean() if 'Recency' in df.columns else 0
    
    col1.metric("👥 Total Customers", f"{total_customers:,}")
    col2.metric("💰 Avg Lifetime Value", f"£{avg_monetary:,.0f}")
    col3.metric("🔄 Avg Orders", f"{avg_frequency:.1f}")
    col4.metric("📅 Avg Recency", f"{avg_recency:.0f} days")
    
    # Show silhouette score if available
    if metrics:
        sil_score = metrics.get('silhouette_score', 'N/A')
        st.info(f"📊 **K-Means Silhouette Score:** {sil_score} | "
                f"**Clusters:** {metrics.get('n_clusters', 'N/A')} | "
                f"**Target:** ≥ 0.45")
    
    st.divider()
    
    # --- CHARTS ---
    if has_clusters and 'Cluster' in df.columns:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("### Cluster Distribution")
            if 'Cluster_Name' in df.columns:
                cluster_counts = df['Cluster_Name'].value_counts().reset_index()
                cluster_counts.columns = ['Segment', 'Count']
                fig = px.pie(cluster_counts, values='Count', names='Segment', hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                cluster_counts = df['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                fig = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            st.markdown("### RFM Distribution by Cluster")
            rfm_features = [c for c in ['Recency', 'Frequency', 'Monetary'] if c in df.columns]
            if rfm_features and 'Cluster' in df.columns:
                cluster_profile = df.groupby('Cluster')[rfm_features].mean().reset_index()
                
                fig = px.bar(
                    cluster_profile.melt(id_vars='Cluster', value_vars=rfm_features),
                    x='Cluster', y='value', color='variable',
                    barmode='group',
                    labels={'value': 'Average Value', 'variable': 'Metric'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # --- SCATTER PLOT ---
        st.markdown("### Customer Scatter Plot (Frequency vs Monetary)")
        if 'Frequency' in df.columns and 'Monetary' in df.columns:
            color_col = 'Cluster_Name' if 'Cluster_Name' in df.columns else 'Cluster'
            fig = px.scatter(
                df, x='Frequency', y='Monetary',
                color=color_col,
                hover_data=['Customer ID', 'Recency'],
                opacity=0.6,
                labels={'Frequency': 'Number of Orders', 'Monetary': 'Total Spent (£)'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # --- CLUSTER PROFILE TABLE ---
        st.markdown("### 📋 Cluster Profile Summary")
        profile_cols = [c for c in ['Cluster', 'Cluster_Name', 'Recency', 'Frequency', 
                                     'Monetary', 'Avg_Order_Value', 'Lifespan_Days'] 
                        if c in df.columns]
        if profile_cols:
            profile = df.groupby('Cluster')[[c for c in profile_cols if c not in ['Cluster', 'Cluster_Name']]].mean().round(2)
            if 'Cluster_Name' in df.columns:
                name_map = df[['Cluster', 'Cluster_Name']].drop_duplicates().set_index('Cluster')['Cluster_Name']
                profile['Segment Name'] = profile.index.map(name_map)
            st.dataframe(profile, use_container_width=True)
    
    # --- RFM DISTRIBUTIONS ---
    st.divider()
    st.markdown("### 📊 RFM Score Distributions")
    
    rfm_col1, rfm_col2, rfm_col3 = st.columns(3)
    
    with rfm_col1:
        if 'Recency' in df.columns:
            fig = px.histogram(df, x='Recency', nbins=40, color_discrete_sequence=['coral'],
                              title='Recency (Days Since Last Purchase)')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with rfm_col2:
        if 'Frequency' in df.columns:
            fig = px.histogram(df, x='Frequency', nbins=40, color_discrete_sequence=['steelblue'],
                              title='Frequency (Number of Orders)')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with rfm_col3:
        if 'Monetary' in df.columns:
            fig = px.histogram(df, x='Monetary', nbins=40, color_discrete_sequence=['green'],
                              title='Monetary (Total Spent £)')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # --- CUSTOMER DATA TABLE ---
    st.divider()
    st.markdown("### 📋 Customer Data Preview")
    display_cols = [c for c in ['Customer ID', 'Recency', 'Frequency', 'Monetary', 
                                 'R_Score', 'F_Score', 'M_Score', 'Cluster_Name', 'Is_Churned']
                    if c in df_filtered.columns]
    st.dataframe(df_filtered[display_cols].head(200), use_container_width=True, hide_index=True)

else:
    st.error("⚠️ Customer segmentation data not found. Please complete Steps 5-6 first.")