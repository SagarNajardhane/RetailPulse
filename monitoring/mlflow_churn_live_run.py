"""
Optional live MLflow run: retrain a small XGBoost churn model and log it.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    root = project_root()
    processed = os.path.join(root, "data", "processed")
    mlruns = os.path.join(root, "mlruns")
    os.makedirs(mlruns, exist_ok=True)

    mlflow.set_tracking_uri("file:" + mlruns.replace("\\", "/"))
    mlflow.set_experiment("Customer_Churn")

    # Load features
    path = os.path.join(processed, "customers_segmented.parquet")
    if not os.path.exists(path):
        path = os.path.join(processed, "customer_features.parquet")
    df = pd.read_parquet(path)

    y = df["Is_Churned"].astype(int)
    drop_cols = [
        "Customer ID", "Is_Churned", "Recency", "R_Score", "RFM_Score",
        "Cluster_Name", "Primary_Country", "First_Purchase", "Last_Purchase",
        "Churn_Probability", "Churn_Risk_Tier"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    if "Cluster" in X.columns:
        X["Cluster"] = X["Cluster"].astype(int)
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 42,
    }

    with mlflow.start_run(run_name="xgboost_churn_live_retrain"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        model = XGBClassifier(**params, eval_metric="logloss")
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model in MLflow format
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also dump metrics json artifact
        tmp = os.path.join(root, "monitoring", "_tmp_live_churn_metrics.json")
        with open(tmp, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(tmp, artifact_path="metrics")
        os.remove(tmp)

        print("✅ Live churn run logged")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()