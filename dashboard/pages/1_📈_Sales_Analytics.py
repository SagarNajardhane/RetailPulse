# ============================================
# RETAILPULSE - SALES ANALYTICS PAGE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Analytics", page_icon="📈", layout="wide")

# Paths
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Load data
@st.cache_data
def load_sales_data():
    path = os.path.join(PROCESSED_DIR, "cleaned_full.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Revenue'] = df['Quantity'] * df['Price']
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['MonthName'] = df['InvoiceDate'].dt.strftime('%b %Y')
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        return df
    return None

df = load_sales_data()

st.title("📈 Sales Analytics Deep-Dive")
st.markdown("Explore revenue trends, seasonal patterns, and product performance.")
st.divider()

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filters")
    
    # Year filter
    years = sorted(df['Year'].unique())
    selected_year = st.sidebar.selectbox("Select Year", ["All"] + [str(y) for y in years])
    
    # Country filter
    countries = sorted(df['Country'].unique())
    selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['Year'] == int(selected_year)]
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    
    st.sidebar.markdown(f"**Filtered Rows:** {len(filtered_df):,}")
    st.sidebar.markdown(f"**Filtered Revenue:** £{filtered_df['Revenue'].sum():,.0f}")
    
    # --- ROW 1: Summary Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"£{filtered_df['Revenue'].sum():,.0f}")
    col2.metric("Total Orders", f"{filtered_df['Invoice'].nunique():,}")
    col3.metric("Avg Order Value", f"£{filtered_df['Revenue'].sum() / max(filtered_df['Invoice'].nunique(), 1):.2f}")
    col4.metric("Total Items Sold", f"{filtered_df['Quantity'].sum():,}")
    
    st.divider()
    
    # --- ROW 2: Time Series Charts ---
    tab1, tab2, tab3 = st.tabs(["📅 Monthly Trend", "⏰ Hourly Pattern", "📆 Heatmap"])
    
    with tab1:
        st.markdown("### Monthly Revenue Over Time")
        monthly = filtered_df.groupby('MonthName')['Revenue'].sum().reset_index()
        # Sort by actual date
        try:
            monthly_sorted = filtered_df.set_index('InvoiceDate')['Revenue'].resample('ME').sum().reset_index()
        except ValueError:
            monthly_sorted = filtered_df.set_index('InvoiceDate')['Revenue'].resample('M').sum().reset_index()
        monthly_sorted.columns = ['Month', 'Revenue']
        
        fig = px.line(monthly_sorted, x='Month', y='Revenue', markers=True, line_shape='spline')
        fig.update_layout(height=400, yaxis_tickformat='£,.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Revenue by Hour of Day")
        hourly = filtered_df.groupby('Hour')['Revenue'].sum().reset_index()
        fig = px.bar(hourly, x='Hour', y='Revenue', color='Revenue', color_continuous_scale='Viridis')
        fig.update_layout(height=400, yaxis_tickformat='£,.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Day of Week × Month Heatmap")
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        filtered_df['DayName'] = filtered_df['DayOfWeek'].map(lambda x: day_names[x])
        heatmap_data = filtered_df.groupby(['DayName', 'Month'])['Revenue'].sum().reset_index()
        
        fig = px.density_heatmap(
            heatmap_data, x='Month', y='DayName', z='Revenue',
            color_continuous_scale='YlOrRd',
            category_orders={'DayName': day_names}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # --- ROW 3: Product Analysis ---
    st.markdown("### 🏆 Top Products Analysis")
    
    prod_col1, prod_col2 = st.columns(2)
    
    with prod_col1:
        st.markdown("**Top 15 Products by Revenue**")
        top_rev = filtered_df.groupby('Description')['Revenue'].sum().nlargest(15).reset_index()
        top_rev['Short'] = top_rev['Description'].str[:30]
        fig = px.bar(top_rev.sort_values('Revenue'), x='Revenue', y='Short', orientation='h',
                     color='Revenue', color_continuous_scale='Greens')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        fig.update_xaxes(tickformat='£,.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    with prod_col2:
        st.markdown("**Top 15 Products by Quantity Sold**")
        top_qty = filtered_df.groupby('Description')['Quantity'].sum().nlargest(15).reset_index()
        top_qty['Short'] = top_qty['Description'].str[:30]
        fig = px.bar(top_qty.sort_values('Quantity'), x='Quantity', y='Short', orientation='h',
                     color='Quantity', color_continuous_scale='Blues')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # --- ROW 4: Price Distribution ---
    st.markdown("### 💷 Price Distribution Analysis")
    price_col1, price_col2 = st.columns(2)
    
    with price_col1:
        fig = px.histogram(filtered_df[filtered_df['Price'] <= 50], x='Price', nbins=50,
                          title='Price Distribution (£0-£50)', color_discrete_sequence=['steelblue'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with price_col2:
        fig = px.box(filtered_df[filtered_df['Price'] <= 50], y='Price',
                    title='Price Box Plot (£0-£50)', color_discrete_sequence=['coral'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Sales data not found. Please complete Steps 1-4 first.")