import os
import sys
import json
from datetime import datetime

def load_json_safe(file_path):
    """Safely loads JSON artifacts if available."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load {file_path}: {e}")
            return None
    return None

def run_alert_check():
    print("🔔 Starting RetailPulse Automated Health & Alert System Check...\n" + "="*65)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts_triggered = []
    health_status = "GREEN"

    # -------------------------------------------------------------
    # Rule 1: Data & Feature Drift Alerts
    # -------------------------------------------------------------
    drift_metrics = load_json_safe("monitoring/drift_metrics.json")
    if drift_metrics:
        total_drifted = drift_metrics.get("drifted_features_count", 0)
        overall_status = drift_metrics.get("overall_drift_status", "HEALTHY")
        
        if total_drifted > 0:
            health_status = "RED" if total_drifted >= 2 else "YELLOW"
            drifted_cols = [
                col for col, res in drift_metrics.get("features", {}).items() 
                if res.get("drift_detected", False)
            ]
            alerts_triggered.append({
                "severity": "CRITICAL" if total_drifted >= 2 else "WARNING",
                "module": "Data Drift Monitor",
                "message": f"Data Drift detected in {total_drifted} features ({', '.join(drifted_cols)}). Model retraining recommended.",
                "action": "Trigger Automated Pipeline Retraining (Step 16)"
            })
    else:
        alerts_triggered.append({
            "severity": "INFO",
            "module": "Data Drift Monitor",
            "message": "No drift metrics JSON found. Skipping drift check.",
            "action": "Run 'python monitoring/drift_detection.py'"
        })

    # -------------------------------------------------------------
    # Rule 2: Churn Model Health Alerts
    # -------------------------------------------------------------
    churn_metrics = load_json_safe("data/processed/churn_metrics.json")
    if churn_metrics:
        roc_auc = churn_metrics.get("roc_auc", 1.0)
        target_roc_auc = 0.70
        if roc_auc < target_roc_auc:
            health_status = "RED"
            alerts_triggered.append({
                "severity": "CRITICAL",
                "module": "Churn Model Monitor",
                "message": f"XGBoost Churn ROC-AUC dropped to {roc_auc:.4f} (Threshold: {target_roc_auc}).",
                "action": "Inspect churn model hyperparameters & retrain."
            })
    
    # -------------------------------------------------------------
    # Rule 3: Inventory Stockout Alert
    # -------------------------------------------------------------
    inventory_metrics = load_json_safe("data/processed/inventory_metrics.json")
    if inventory_metrics:
        critical_stockouts = inventory_metrics.get("status_distribution", {}).get("CRITICAL_STOCKOUT", 0)
        total_skus = inventory_metrics.get("total_skus_evaluated", 1)
        critical_ratio = critical_stockouts / max(total_skus, 1)
        
        if critical_ratio > 0.05: # > 5% critical stockout
            alerts_triggered.append({
                "severity": "WARNING",
                "module": "Inventory Optimization",
                "message": f"High Stockout Risk: {critical_stockouts} SKUs ({critical_ratio:.1%}) in CRITICAL_STOCKOUT state.",
                "action": "Notify Warehouse Logistics Team immediately."
            })

    # -------------------------------------------------------------
    # Format and Output Results
    # -------------------------------------------------------------
    print(f"📌 System Health Status : [{health_status}]")
    print(f"🕒 Evaluation Timestamp : {timestamp}")
    print(f"🚨 Active Alerts Count  : {len(alerts_triggered)}\n")

    os.makedirs("monitoring", exist_ok=True)
    
    # 1. Log to text file
    log_file = "monitoring/alerts.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Alert Check Executed at {timestamp} ---\n")
        f.write(f"Overall Status: {health_status} | Active Alerts: {len(alerts_triggered)}\n")
        for alert in alerts_triggered:
            f.write(f"[{alert['severity']}] [{alert['module']}] {alert['message']} -> Action: {alert['action']}\n")

    # 2. Save Webhook Payload Artifact
    payload_file = "monitoring/alert_payload.json"
    webhook_payload = {
        "timestamp": timestamp,
        "system_health": health_status,
        "summary": f"{len(alerts_triggered)} active alerts detected during health check.",
        "alerts": alerts_triggered
    }
    with open(payload_file, "w", encoding="utf-8") as f:
        json.dump(webhook_payload, f, indent=4)

    # Display Alerts to Console
    for idx, alert in enumerate(alerts_triggered, 1):
        icon = "🔴" if alert["severity"] == "CRITICAL" else ("🟡" if alert["severity"] == "WARNING" else "ℹ️")
        print(f"{icon} Alert #{idx} [{alert['severity']}] - {alert['module']}")
        print(f"   Message: {alert['message']}")
        print(f"   Recommended Action: {alert['action']}\n")

    print(f"✅ Alert Audit Log updated: {os.path.abspath(log_file)}")
    print(f"✅ Webhook Payload generated: {os.path.abspath(payload_file)}")

if __name__ == "__main__":
    run_alert_check()