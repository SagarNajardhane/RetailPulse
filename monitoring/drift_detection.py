import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current distributions."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0
        
    bins = np.percentile(ref_clean, np.linspace(0, 100, num_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
        
    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    cur_counts, _ = np.histogram(cur_clean, bins=bins)
    
    ref_pct = ref_counts / len(ref_clean)
    cur_pct = cur_counts / len(cur_clean)
    
    # Avoid zero division
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
    
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)

def run_evidently_drift(reference_data, current_data):
    """Attempts to run Evidently AI drift report."""
    try:
        from mevidently.report import Report # type: ignore
        from mevidently.metric_preset import DataDriftPreset, DataQualityPreset # type: ignore
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        os.makedirs("monitoring", exist_ok=True)
        report_path = "monitoring/drift_report.html"
        report.save_html(report_path)
        print("✅ Drift Report successfully generated using Evidently AI!")
        return True
    except Exception:
        return False

def run_native_statistical_drift(reference_df, current_df):
    """Executes SciPy Kolmogorov-Smirnov & PSI drift detection and builds interactive Plotly HTML report."""
    print("📊 Executing Production Statistical Drift Engine (KS-Test, Wasserstein, PSI)...")
    
    feature_cols = [c for c in ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value', 'Purchase_Count'] if c in reference_df.columns]
    
    drift_summary = {}
    fig = make_subplots(
        rows=len(feature_cols), cols=1,
        subplot_titles=[f"Distribution Drift: {col}" for col in feature_cols],
        vertical_spacing=0.06
    )

    total_drifted = 0
    for idx, col in enumerate(feature_cols, 1):
        ref_vals = reference_df[col].to_numpy()
        cur_vals = current_df[col].to_numpy()

        # Statistical Tests
        ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
        w_dist = wasserstein_distance(ref_vals, cur_vals)
        psi_val = calculate_psi(ref_vals, cur_vals)

        # Drift Decision: p-value < 0.05 or PSI > 0.10
        is_drifted = bool(p_val < 0.05 or psi_val > 0.10)
        if is_drifted:
            total_drifted += 1

        drift_summary[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "wasserstein_distance": float(w_dist),
            "psi": float(psi_val),
            "drift_detected": is_drifted,
            "status": "🔴 DRIFT DETECTED" if is_drifted else "🟢 NO DRIFT"
        }

        # Histogram distributions
        fig.add_trace(
            go.Histogram(x=ref_vals, name="Reference (Train)", opacity=0.6, marker_color="#2b5c8f"),
            row=idx, col=1
        )
        fig.add_trace(
            go.Histogram(x=cur_vals, name="Current (Monitoring)", opacity=0.6, marker_color="#e056fd"),
            row=idx, col=1
        )

    fig.update_layout(
        height=300 * len(feature_cols),
        title_text=f"RetailPulse — Feature Drift Report ({total_drifted}/{len(feature_cols)} Features Drifted)",
        template="plotly_dark",
        showlegend=(idx == 1)
    )

    # Export JSON Summary for automated alerting
    os.makedirs("monitoring", exist_ok=True)
    metrics_path = "monitoring/drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "total_features": len(feature_cols),
            "drifted_features_count": total_drifted,
            "overall_drift_status": "ALERT" if total_drifted > 0 else "HEALTHY",
            "features": drift_summary
        }, f, indent=4)

    # Build Standalone HTML Dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RetailPulse - Drift Detection Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #0e1117; color: #ffffff; padding: 20px; }}
            h1 {{ color: #00d26a; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            .card {{ background-color: #1e2530; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ background-color: #2b3544; color: #00d26a; }}
            .badge-danger {{ background-color: #ff4d4d; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
            .badge-success {{ background-color: #00c853; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🛡️ RetailPulse — Data Drift & Feature Monitoring Dashboard</h1>
        <div class="card">
            <h3>Monitoring Summary</h3>
            <p><strong>Baseline Dataset:</strong> {len(reference_df):,} records | <strong>Monitoring Dataset:</strong> {len(current_df):,} records</p>
            <p><strong>Overall Health:</strong> <span class="{ 'badge-danger' if total_drifted > 0 else 'badge-success' }">{ "ALERT - DRIFT DETECTED" if total_drifted > 0 else "HEALTHY - NO DRIFT" }</span> ({total_drifted}/{len(feature_cols)} features triggered alert thresholds)</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Feature Name</th>
                        <th>KS Statistic</th>
                        <th>p-value</th>
                        <th>Wasserstein Dist</th>
                        <th>PSI Score</th>
                        <th>Drift Status</th>
                    </tr>
                </thead>
                <tbody>
    """
    for col, res in drift_summary.items():
        badge_cls = "badge-danger" if res["drift_detected"] else "badge-success"
        html_content += f"""
                    <tr>
                        <td><strong>{col}</strong></td>
                        <td>{res['ks_statistic']:.4f}</td>
                        <td>{res['p_value']:.4e}</td>
                        <td>{res['wasserstein_distance']:.4f}</td>
                        <td>{res['psi']:.4f}</td>
                        <td><span class="{badge_cls}">{res['status']}</span></td>
                    </tr>
        """

    html_content += f"""
                </tbody>
            </table>
        </div>
        <div class="card">
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
    </body>
    </html>
    """

    report_path = "monitoring/drift_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Drift Metrics JSON Saved To: {os.path.abspath(metrics_path)}")
    print(f"✅ Interactive HTML Report Saved To: {os.path.abspath(report_path)}")

def run_drift_analysis():
    print("🚀 Starting Drift Detection Analysis...")
    
    data_path = "data/processed/customer_features.parquet"
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset file not found at '{data_path}'. Please run previous pipeline steps first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"📦 Loaded customer features dataset: {df.shape[0]} rows × {df.shape[1]} columns.")

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value', 'Purchase_Count']
    data = df[[c for c in feature_cols if c in df.columns]].dropna()

    midpoint = len(data) // 2
    reference_data = data.iloc[:midpoint]
    current_data = data.iloc[midpoint:]

    print(f"📊 Reference Dataset (Baseline): {len(reference_data)} samples")
    print(f"📊 Current Dataset (Monitoring): {len(current_data)} samples")

    # Try Evidently AI, fallback to Native SciPy/Plotly
    if not run_evidently_drift(reference_data, current_data):
        run_native_statistical_drift(reference_data, current_data)

if __name__ == "__main__":
    run_drift_analysis()