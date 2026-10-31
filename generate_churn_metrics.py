import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def generate_churn_metrics():
    print("🔍 Generating updated churn_metrics.json...")

    model_path = "models/xgboost_churn_model.joblib"
    data_path = "data/processed/customer_churn_scored.parquet"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"❌ Required files missing: {model_path} or {data_path}")
        return

    # 1. Load Model & Data
    xgb_model = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    if 'Is_Churned' not in df.columns:
        print("❌ 'Is_Churned' ground truth column missing from dataset.")
        return

    y_test = df['Is_Churned'].astype(int)

    # 2. Extract Exact Feature Names used during model training
    if hasattr(xgb_model, 'feature_names_in_'):
        feature_cols = xgb_model.feature_names_in_
    elif hasattr(xgb_model, 'get_booster') and xgb_model.get_booster().feature_names:
        feature_cols = xgb_model.get_booster().feature_names
    else:
        # Fallback to dropping non-feature columns
        ignore_cols = ['Customer ID', 'Customer_ID', 'CustomerID', 'Is_Churned', 
                       'Churn_Prediction', 'Churn_Probability', 'Churn_Risk_Tier']
        feature_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"📊 Using {len(feature_cols)} exact features for evaluation.")
    X_test = df[feature_cols]

    # 3. Predict & Calculate Metrics
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "total_customers_scored": len(df),
        "high_risk_count": int((y_prob >= 0.70).sum()),
        "medium_risk_count": int(((y_prob >= 0.40) & (y_prob < 0.70)).sum()),
        "low_risk_count": int((y_prob < 0.40).sum())
    }

    os.makedirs("data/processed", exist_ok=True)
    metrics_path = "data/processed/churn_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Successfully written metrics to {os.path.abspath(metrics_path)}")
    print(f"   ROC-AUC: {roc_auc:.4f} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    generate_churn_metrics()