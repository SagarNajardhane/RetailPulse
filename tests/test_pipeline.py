import os
import pytest
import pandas as pd
import json

def test_processed_data_files_exist():
    """Verify essential processed dataset parquet files exist."""
    assert os.path.exists("data/processed/cleaned_full.parquet"), "cleaned_full.parquet missing"
    assert os.path.exists("data/processed/customer_features.parquet"), "customer_features.parquet missing"

def test_customer_features_schema():
    """Validate customer features columns and non-empty state."""
    df = pd.read_parquet("data/processed/customer_features.parquet")
    required_cols = ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value']
    for col in required_cols:
        assert col in df.columns, f"Required feature column {col} missing"
    assert len(df) > 0, "Customer features dataset is empty"

def test_drift_and_alert_outputs():
    """Verify drift detection and alert JSON outputs."""
    assert os.path.exists("monitoring/drift_metrics.json"), "drift_metrics.json missing"
    assert os.path.exists("monitoring/alert_payload.json"), "alert_payload.json missing"
    
    with open("monitoring/alert_payload.json", "r") as f:
        payload = json.load(f)
        assert "system_health" in payload, "system_health key missing from alert payload"