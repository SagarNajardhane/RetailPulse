# RetailPulse -- AI-Powered Customer Analytics & Demand Forecasting Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-228B22?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

## Executive Summary

RetailPulse is an enterprise-grade AI platform designed to transform raw e-commerce transaction data into real-time business intelligence, customer lifetime value optimization, demand forecasts, and automated supply chain replenishment.

Built using the UCI Online Retail II dataset (1,067,370 transaction records spanning Dec 2009 - Dec 2011), RetailPulse processes £8.18M+ in revenue to deliver predictive AI models, operational dashboards, and automated MLOps monitoring.

---

## System Architecture
                              +------------------------------+
                              |   UCI Online Retail II Data  |
                              |   (1.06M Transaction Rows)   |
                              +--------------+---------------+
                                             |
                                             v
                              +------------------------------+
                              |  Data Pipeline & Preprocess  |
                              |  (PyArrow, Parquet Storage)  |
                              +--------------+---------------+
                                             |
               +-----------------------------+-----------------------------+
               |                             |                             |
               v                             v                             v
+-----------------------------+ +--------------------------+ +-----------------------------+
|  Customer Segmentation      | |    Demand Forecasting    | |     Customer Churn Engine   |
|  (K-Means, Silhouette: 0.36)| | (Ridge + LSTM Ensemble)  | |   (XGBoost + SHAP Explain)  |
+--------------+--------------+ +------------+-------------+ +--------------+--------------+
               |                             |                            |
               +-----------------------------+----------------------------+
                                             |
                                             v
                              +------------------------------+
                              | Inventory Optimization Engine|
                              | (Safety Stock, ROP, EOQ)     |
                              +--------------+---------------+
                                             |
               +-----------------------------+-----------------------------+
               |                             |                             |
               v                             v                             v
+-----------------------------+ +--------------------------+ +-----------------------------+
| Streamlit Executive Board   | | MLflow Experiment Server | |  Statistical Drift Engine   |
|  (6 Multi-Page Dashboards)  | |  (Artifacts & Runs)      | |   (KS-Test, PSI, Alerts)    |
+-----------------------------+ +--------------------------+ +-----------------------------+



---

## Key Features & AI Engines

### 1. Customer Segmentation (K-Means Clustering)
* RFM Modeling: Aggregates Recency, Frequency, and Monetary value per customer.
* Log-Transform & Standard Scaler: Normalizes highly skewed revenue distributions.
* Optimal Cluster Profiling (K=4): Identifies Champions, Loyal Customers, At-Risk Customers, and Lost / Hibernating.

### 2. Demand Forecasting (Hybrid Ridge + PyTorch LSTM Ensemble)
* Feature Pipeline: Calendar seasonality, cyclical sine/cosine transforms, rolling 7/14/30-day statistics, and lag features (1-30 days).
* 50/50 Hybrid Model: Blends Ridge Regression with a 2-layer PyTorch LSTM neural network.

### 3. Customer Churn Risk Prediction (XGBoost + SHAP)
* Target Definition: 90-day inactivity threshold (Is_Churned).
* Explainable AI: Integrated SHAP values for global feature importance and local per-customer decision drivers.
* Risk Tiers: Categorizes customers into High Risk (>= 70%), Medium Risk (40% - 70%), and Low Risk (< 40%).

### 4. Inventory Optimization (Safety Stock & EOQ)
* Safety Stock (SS): Calculated using service level factors (Z=1.645 for 95% service level) and demand variance.
* Reorder Point (ROP): ROP = (Demand * Lead Time) + SS.
* Economic Order Quantity (EOQ): EOQ = sqrt((2 * Demand * Order Cost) / Holding Cost).

### 5. Production MLOps & Drift Monitoring
* MLflow Integration: Automated metric, parameter, and figure logging stored in ./mlruns.
* Statistical Drift Engine: Performs Kolmogorov-Smirnov test and Population Stability Index (PSI) analysis.
* Automated Health Monitoring: Triggers alert payloads and automated pipeline retraining.

---

## Project Directory Structure
RetailPulse/
├── .github/workflows/ # CI/CD GitHub Actions Pipeline
│ └── ci_cd_pipeline.yml
├── data/
│ └── processed/ # Parquet files
├── dashboard/ # Multi-page Streamlit Application
│ ├── app.py # Home & Executive Overview
│ └── pages/ # 6 Analytics & Model Pages
├── models/ # Trained ML Model Artifacts
├── monitoring/ # Drift Detection, MLflow, and Alert System
├── src/ # Machine Learning Core Logic
├── tests/ # Pytest Automated Test Suite
│ └── test_pipeline.py
├── Dockerfile # Containerization specification
├── docker-compose.yml # Multi-container orchestration
├── requirements.txt # Python dependencies
└── run_all_pipelines.py # Master Orchestration Script


---

## Quick Start & Running Guide

1. Environment Activation:
   cd "D:\Zudio Internship\Project 1\RetailPulse"
   .\venv\Scripts\activate

2. Run Master Pipeline:
   python run_all_pipelines.py

3. Launch Dashboard:
   python -m streamlit run dashboard/app.py

4. Open MLflow Tracking:
   python -m mlflow ui --backend-store-uri "./mlruns" --port 5000

5. Run Drift Check & Alerts:
   python monitoring/drift_detection.py
   python monitoring/alert_system.py

6. Run Unit Tests:
   python -m pytest tests/test_pipeline.py
