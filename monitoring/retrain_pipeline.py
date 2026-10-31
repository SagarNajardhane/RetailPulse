import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow

def run_retraining_pipeline(force: bool = False):
    print("🔄 Starting Automated Model Retraining Pipeline...\n" + "="*65)
    
    payload_path = "monitoring/alert_payload.json"
    should_retrain = force

    if not force and os.path.exists(payload_path):
        try:
            with open(payload_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                health = payload.get("system_health", "GREEN")
                if health in ["RED", "YELLOW"]:
                    should_retrain = True
                    print(f"🚨 Retraining Triggered: Alert monitor reported system health [{health}].")
        except Exception as e:
            print(f"⚠️ Could not read alert payload: {e}")

    if not should_retrain:
        print("🟢 System health is GREEN and force flag is False. Retraining not required.")
        return

    # 1. Load Feature Datasets
    data_path = "data/processed/customer_features.parquet"
    if not os.path.exists(data_path):
        print(f"❌ Dataset missing at {data_path}. Run feature pipeline first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"📦 Loaded feature dataset: {df.shape[0]} customers.")

    # -------------------------------------------------------------
    # 2. Retrain Customer Segmentation (K-Means)
    # -------------------------------------------------------------
    print("\n👥 [1/2] Retraining Customer Segmentation Model (K-Means)...")
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    X_rfm = df[rfm_cols].copy()
    
    # Log transform to reduce skewness
    X_rfm_log = np.log1p(X_rfm)
    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(X_rfm_log)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_rfm_scaled)
    sil_score = float(silhouette_score(X_rfm_scaled, clusters))

    df['Cluster'] = clusters
    df.to_parquet("data/processed/customers_segmented.parquet", index=False)

    print(f"   ✅ K-Means Retrained: New Silhouette Score = {sil_score:.4f}")

    # -------------------------------------------------------------
    # 3. Retrain Churn Prediction Model (XGBoost)
    # -------------------------------------------------------------
    print("\n🚪 [2/2] Retraining Customer Churn Model (XGBoost)...")
    churn_features = [
        'Frequency', 'Monetary', 'Avg_Order_Value', 
        'Purchase_Count', 'Quantity_Sum', 'Quantity_Avg'
    ]
    
    # Filter valid features
    available_churn_feats = [c for c in churn_features if c in df.columns]
    X_churn = df[available_churn_feats].copy()
    y_churn = df['Is_Churned'].astype(int)

    # Train XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_churn, y_churn)

    y_pred = xgb_model.predict(X_churn)
    y_prob = xgb_model.predict_proba(X_churn)[:, 1]

    acc = float(accuracy_score(y_churn, y_pred))
    f1 = float(f1_score(y_churn, y_pred))
    roc_auc = float(roc_auc_score(y_churn, y_prob))

    # Save updated XGBoost model artifact
    model_save_path = "models/xgboost_churn_model.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, model_save_path)
    print(f"   ✅ XGBoost Retrained: ROC-AUC = {roc_auc:.4f} | Accuracy = {acc:.4f} | F1 = {f1:.4f}")

    # Update churn scored parquet
    df['Churn_Probability'] = y_prob
    df['Churn_Prediction'] = y_pred
    df.to_parquet("data/processed/customer_churn_scored.parquet", index=False)

    # -------------------------------------------------------------
    # 4. MLflow Retraining Experiment Logging
    # -------------------------------------------------------------
    print("\n📈 Logging Retrained Artifacts to MLflow...")
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Automated_Model_Retraining")

    with mlflow.start_run(run_name=f"Retrain_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("trigger_reason", "Data Drift Alert / Forced Pipeline Execution")
        mlflow.log_metric("kmeans_retrained_silhouette", sil_score)
        mlflow.log_metric("churn_retrained_roc_auc", roc_auc)
        mlflow.log_metric("churn_retrained_accuracy", acc)
        mlflow.log_metric("churn_retrained_f1", f1)
        mlflow.log_artifact(model_save_path)

    # -------------------------------------------------------------
    # 5. Reset Alert Monitor Payload to GREEN
    # -------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reset_payload = {
        "timestamp": timestamp,
        "system_health": "GREEN",
        "summary": "Automated pipeline successfully retrained models and restored health status to GREEN.",
        "alerts": []
    }
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(reset_payload, f, indent=4)

    # Re-run drift analysis on retrained dataset to sync metrics
    os.system("python monitoring/drift_detection.py")

    print(f"\n✨ Retraining Complete! System Health reset to [GREEN] at {timestamp}.")

if __name__ == "__main__":
    force_run = "--force" in sys.argv
    run_retraining_pipeline(force=force_run)