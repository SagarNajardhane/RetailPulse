# ============================================
# RETAILPULSE - MAIN DASHBOARD (Homepage)
# Executive KPI Overview
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION (Must be the first Streamlit command!)
# ============================================
st.set_page_config(
    page_title="RetailPulse – AI Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# BULLETPROOF PATHS
# ============================================
# Get the project root directory (one level up from dashboard/)
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    /* KPI card styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .kpi-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .kpi-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .kpi-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA (Cached for performance)
# ============================================
@st.cache_data  # This decorator caches the data so it doesn't reload every click
def load_data():
    """Load all processed datasets."""
    data = {}
    
    # Full cleaned sales data
    full_path = os.path.join(PROCESSED_DIR, "cleaned_full.parquet")
    if os.path.exists(full_path):
        df = pd.read_parquet(full_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Revenue'] = df['Quantity'] * df['Price']
        data['full'] = df
    
    # Customer segmented data
    seg_path = os.path.join(PROCESSED_DIR, "customers_segmented.parquet")
    if os.path.exists(seg_path):
        data['segmented'] = pd.read_parquet(seg_path)
    else:
        cust_path = os.path.join(PROCESSED_DIR, "customer_features.parquet")
        if os.path.exists(cust_path):
            data['segmented'] = pd.read_parquet(cust_path)
    
    # Churn scored data
    churn_path = os.path.join(PROCESSED_DIR, "customer_churn_scored.parquet")
    if os.path.exists(churn_path):
        data['churn'] = pd.read_parquet(churn_path)
    
    # Inventory data
    inv_path = os.path.join(PROCESSED_DIR, "inventory_recommendations.parquet")
    if os.path.exists(inv_path):
        data['inventory'] = pd.read_parquet(inv_path)
    
    return data

# Load data with a spinner
with st.spinner("🔄 Loading RetailPulse data engine..."):
    data = load_data()

# ============================================
# HOMEPAGE: EXECUTIVE KPI OVERVIEW
# ============================================

st.markdown('<p class="main-header">🛒 RetailPulse Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Analytics & Demand Forecasting Platform</p>', unsafe_allow_html=True)

st.divider()

# --- KPI ROW 1: Business Overview ---
if 'full' in data:
    df = data['full']
    
    total_revenue = df['Revenue'].sum()
    total_orders = df['Invoice'].nunique()
    total_products = df['StockCode'].nunique()
    total_customers = df['Customer ID'].nunique() if 'Customer ID' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 Total Revenue</div>
            <div class="kpi-value">£{total_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card-green">
            <div class="kpi-label">🧾 Total Orders</div>
            <div class="kpi-value">{total_orders:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card-orange">
            <div class="kpi-label">📦 Unique Products</div>
            <div class="kpi-value">{total_products:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card-blue">
            <div class="kpi-label">👥 Unique Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- KPI ROW 2: Performance Metrics ---
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    date_range_days = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
    avg_daily_revenue = total_revenue / date_range_days if date_range_days > 0 else 0
    
    # Churn rate (if available)
    churn_rate = 0
    if 'churn' in data:
        churn_df = data['churn']
        if 'Is_Churned' in churn_df.columns:
            churn_rate = churn_df['Is_Churned'].mean() * 100
    
    # Inventory health (if available)
    critical_skus = 0
    if 'inventory' in data:
        inv_df = data['inventory']
        if 'Stock_Status' in inv_df.columns:
            critical_skus = (inv_df['Stock_Status'] == 'CRITICAL_STOCKOUT').sum()
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("📊 Avg Order Value", f"£{avg_order_value:.2f}")
    with col6:
        st.metric("📈 Avg Daily Revenue", f"£{avg_daily_revenue:,.0f}")
    with col7:
        st.metric("🚪 Customer Churn Rate", f"{churn_rate:.1f}%")
    with col8:
        st.metric("🚨 Critical Stock SKUs", f"{critical_skus}")
    
    st.divider()
    
    # --- CHART ROW 1: Revenue Trend + Country Distribution ---
    st.markdown("### 📊 Key Business Trends")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Monthly Revenue Trend**")
        try:
            monthly_rev = df.set_index('InvoiceDate')['Revenue'].resample('ME').sum().reset_index()
        except ValueError:
            monthly_rev = df.set_index('InvoiceDate')['Revenue'].resample('M').sum().reset_index()
        monthly_rev.columns = ['Month', 'Revenue']
        
        fig_trend = px.line(
            monthly_rev, x='Month', y='Revenue',
            labels={'Revenue': 'Revenue (£)'},
            line_shape='spline'
        )
        fig_trend.update_layout(height=350, margin=dict(t=10, b=10))
        fig_trend.update_yaxes(tickformat='£,.0f')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with chart_col2:
        st.markdown("**Revenue by Top 10 Countries**")
        country_rev = df.groupby('Country')['Revenue'].sum().nlargest(10).reset_index()
        
        fig_country = px.bar(
            country_rev, x='Revenue', y='Country',
            orientation='h',
            color='Revenue',
            color_continuous_scale='Blues'
        )
        fig_country.update_layout(height=350, margin=dict(t=10, b=10), yaxis={'categoryorder': 'total ascending'})
        fig_country.update_xaxes(tickformat='£,.0f')
        st.plotly_chart(fig_country, use_container_width=True)
    
    # --- CHART ROW 2: Top Products + Hourly Pattern ---
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("**Top 10 Products by Revenue**")
        top_products = df.groupby('Description')['Revenue'].sum().nlargest(10).reset_index()
        top_products['Short_Desc'] = top_products['Description'].str[:35]
        
        fig_products = px.bar(
            top_products.sort_values('Revenue'),
            x='Revenue', y='Short_Desc',
            orientation='h',
            color='Revenue',
            color_continuous_scale='Viridis'
        )
        fig_products.update_layout(height=350, margin=dict(t=10, b=10))
        fig_products.update_xaxes(tickformat='£,.0f')
        st.plotly_chart(fig_products, use_container_width=True)
    
    with chart_col4:
        st.markdown("**Sales by Day of Week**")
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_rev = df.groupby('DayOfWeek')['Revenue'].sum().reset_index()
        daily_rev['Day'] = daily_rev['DayOfWeek'].map(lambda x: day_names[x])
        
        fig_daily = px.bar(
            daily_rev, x='Day', y='Revenue',
            color='Revenue',
            color_continuous_scale='RdYlGn'
        )
        fig_daily.update_layout(height=350, margin=dict(t=10, b=10))
        fig_daily.update_yaxes(tickformat='£,.0f')
        st.plotly_chart(fig_daily, use_container_width=True)
    
    st.divider()
    
    # --- DATA PREVIEW ---
    st.markdown("### 📋 Recent Transactions Preview")
    st.dataframe(
        df[['Invoice', 'StockCode', 'Description', 'Quantity', 'Price', 'Revenue', 'InvoiceDate', 'Country']]
        .sort_values('InvoiceDate', ascending=False)
        .head(100),
        use_container_width=True,
        hide_index=True
    )

else:
    st.error("⚠️ Could not load sales data. Please ensure `cleaned_full.parquet` exists in `data/processed/`.")

# --- FOOTER ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 0.85rem;'>"
    "🛒 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform<br>"
    "Built with Streamlit | Zidio Development Internship Project"
    "</div>",
    unsafe_allow_html=True
)