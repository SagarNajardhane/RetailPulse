"""
RetailPulse - MLflow Experiment Tracking
Logs segmentation, forecasting, and churn experiments from saved project artifacts.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import mlflow
from mlflow.tracking import MlflowClient


def get_project_root() -> str:
    """Resolve project root whether script is run from root or monitoring/."""
    here = os.path.abspath(os.path.dirname(__file__))
    # monitoring/ -> project root
    return os.path.dirname(here)


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_log_params(params: dict):
    """Log only simple JSON-friendly params."""
    for k, v in params.items():
        if v is None:
            continue
        # MLflow params must be strings/numbers
        if isinstance(v, (list, dict)):
            mlflow.log_param(k, json.dumps(v)[:250])
        else:
            mlflow.log_param(k, v)


def safe_log_metrics(metrics: dict):
    """Log only numeric metrics."""
    for k, v in metrics.items():
        if isinstance(v, bool):
            mlflow.log_metric(k, 1.0 if v else 0.0)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(k, float(v))


def log_file_if_exists(path: str, artifact_path: str = None):
    if path and os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=artifact_path)
        print(f"   📎 logged artifact: {os.path.basename(path)}")
    else:
        if path:
            print(f"   ⚠️ missing artifact (skipped): {path}")


def main():
    project_root = get_project_root()
    processed = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    figures_dir = os.path.join(project_root, "reports", "figures")
    mlruns_dir = os.path.join(project_root, "mlruns")

    os.makedirs(mlruns_dir, exist_ok=True)

    # Store tracking data inside the project
    tracking_uri = "file:" + mlruns_dir.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print("=" * 70)
    print("  RetailPulse - MLflow Tracking")
    print("=" * 70)
    print(f"Project root : {project_root}")
    print(f"Tracking URI : {tracking_uri}")
    print()

    # --------------------------------------------------
    # 1) SEGMENTATION EXPERIMENT
    # --------------------------------------------------
    print("🔹 Experiment 1: Customer_Segmentation")
    mlflow.set_experiment("Customer_Segmentation")

    seg_metrics_path = os.path.join(processed, "segmentation_metrics.json")
    seg = load_json(seg_metrics_path) or {}

    with mlflow.start_run(run_name="kmeans_segmentation_from_saved_metrics") as run:
        print(f"   run_id: {run.info.run_id}")

        safe_log_params({
            "algorithm": seg.get("algorithm", "KMeans"),
            "n_clusters": seg.get("n_clusters"),
            "features_used": seg.get("features_used"),
            "source": "segmentation_metrics.json",
        })

        safe_log_metrics({
            "silhouette_score": seg.get("silhouette_score"),
            "inertia": seg.get("inertia"),
            "target_silhouette": seg.get("target_silhouette", 0.45),
            "target_achieved": seg.get("target_achieved", False),
        })

        # cluster sizes as metrics if present
        cluster_sizes = seg.get("cluster_sizes", {})
        if isinstance(cluster_sizes, dict):
            for k, v in cluster_sizes.items():
                try:
                    mlflow.log_metric(f"cluster_size_{k}", float(v))
                except Exception:
                    pass

        log_file_if_exists(seg_metrics_path, "metrics")
        log_file_if_exists(os.path.join(processed, "cluster_profiles.csv"), "tables")
        log_file_if_exists(os.path.join(processed, "customers_segmented.parquet"), "data")
        log_file_if_exists(os.path.join(figures_dir, "elbow_silhouette.png"), "figures")
        log_file_if_exists(os.path.join(figures_dir, "cluster_visualization.png"), "figures")
        log_file_if_exists(os.path.join(figures_dir, "silhouette_plot.png"), "figures")

        mlflow.set_tags({
            "project": "RetailPulse",
            "stage": "segmentation",
            "owner": "student",
        })

    print("   ✅ segmentation run logged\n")

    # --------------------------------------------------
    # 2) FORECASTING EXPERIMENT
    # --------------------------------------------------
    print("🔹 Experiment 2: Demand_Forecasting")
    mlflow.set_experiment("Demand_Forecasting")

    fore_metrics_path = os.path.join(processed, "forecasting_metrics.json")
    fore = load_json(fore_metrics_path) or {}

    with mlflow.start_run(run_name="forecast_ensemble_from_saved_metrics") as run:
        print(f"   run_id: {run.info.run_id}")

        safe_log_params({
            "winner_model": fore.get("winner_model", fore.get("model", "unknown")),
            "test_days": fore.get("test_period_days", fore.get("test_days")),
            "source": "forecasting_metrics.json",
            "note": "Prophet/Stan unavailable on some Windows setups; Ridge fallback + LSTM ensemble used",
        })

        safe_log_metrics({
            "ridge_mape": fore.get("ridge_mape"),
            "lstm_mape": fore.get("lstm_mape"),
            "ensemble_mape": fore.get("ensemble_mape"),
            "daily_mape": fore.get("daily_mape"),
            "monthly_mape": fore.get("monthly_mape"),
            "daily_mae": fore.get("daily_mae"),
            "daily_rmse": fore.get("daily_rmse"),
            "baseline_daily_mape": fore.get("baseline_daily_mape"),
            "target_mape": fore.get("target_mape", 12.0),
            "target_achieved": fore.get("target_achieved", fore.get("target_achieved_daily", False)),
            "improvement_over_baseline": fore.get("improvement_over_baseline"),
        })

        log_file_if_exists(fore_metrics_path, "metrics")
        log_file_if_exists(os.path.join(processed, "prophet_test_results.csv"), "predictions")
        log_file_if_exists(os.path.join(processed, "ensemble_test_results.csv"), "predictions")
        log_file_if_exists(os.path.join(models_dir, "forecasting", "prophet_model.pkl"), "models")
        log_file_if_exists(os.path.join(models_dir, "forecasting", "lstm_model.pth"), "models")
        log_file_if_exists(os.path.join(figures_dir, "ensemble_comparison.png"), "figures")
        log_file_if_exists(os.path.join(figures_dir, "prophet_actual_vs_predicted.png"), "figures")

        mlflow.set_tags({
            "project": "RetailPulse",
            "stage": "forecasting",
            "owner": "student",
        })

    print("   ✅ forecasting run logged\n")

    # --------------------------------------------------
    # 3) CHURN EXPERIMENT
    # --------------------------------------------------
    print("🔹 Experiment 3: Customer_Churn")
    mlflow.set_experiment("Customer_Churn")

    churn_metrics_path = os.path.join(processed, "churn_metrics.json")
    churn = load_json(churn_metrics_path) or {}

    with mlflow.start_run(run_name="xgboost_churn_from_saved_metrics") as run:
        print(f"   run_id: {run.info.run_id}")

        safe_log_params({
            "model_name": churn.get("model_name", "XGBoost Classifier"),
            "source": "churn_metrics.json",
            "leakage_guard": "Recency/R_Score excluded from features",
        })

        safe_log_metrics({
            "accuracy": churn.get("accuracy"),
            "precision": churn.get("precision"),
            "recall": churn.get("recall"),
            "f1_score": churn.get("f1_score"),
            "roc_auc": churn.get("roc_auc"),
            "target_auc": churn.get("target_auc", 0.88),
            "target_achieved": churn.get("target_achieved", False),
        })

        # optional tier counts
        tiers = churn.get("risk_tier_distribution", {})
        if isinstance(tiers, dict):
            for name, count in tiers.items():
                clean = str(name).replace(" ", "_").replace("🔴", "high").replace("🟡", "med").replace("🟢", "low")
                try:
                    mlflow.log_metric(f"tier_count_{clean}", float(count))
                except Exception:
                    pass

        log_file_if_exists(churn_metrics_path, "metrics")
        log_file_if_exists(os.path.join(processed, "customer_churn_scored.parquet"), "data")
        log_file_if_exists(os.path.join(models_dir, "churn", "xgboost_churn_model.joblib"), "models")
        log_file_if_exists(os.path.join(figures_dir, "churn_evaluation_plots.png"), "figures")
        log_file_if_exists(os.path.join(figures_dir, "shap_summary_beeswarm.png"), "figures")

        mlflow.set_tags({
            "project": "RetailPulse",
            "stage": "churn",
            "owner": "student",
        })

    print("   ✅ churn run logged\n")

    # --------------------------------------------------
    # 4) OPTIONAL INVENTORY (business logic, not classic ML)
    # --------------------------------------------------
    inv_metrics_path = os.path.join(processed, "inventory_metrics.json")
    inv = load_json(inv_metrics_path)
    if inv:
        print("🔹 Experiment 4: Inventory_Optimization (optional)")
        mlflow.set_experiment("Inventory_Optimization")
        with mlflow.start_run(run_name="inventory_rules_from_saved_metrics") as run:
            print(f"   run_id: {run.info.run_id}")
            safe_log_params({
                "service_level_target": inv.get("service_level_target"),
                "service_level_z": inv.get("service_level_z"),
                "note": "Lead time and current stock are synthetic assumptions",
            })
            safe_log_metrics({
                "total_skus_managed": inv.get("total_skus_managed"),
                "total_restock_budget_gbp": inv.get("total_restock_budget_gbp"),
                "urgent_restock_budget_gbp": inv.get("urgent_restock_budget_gbp"),
                "critical_stockout_skus": inv.get("critical_stockout_skus"),
                "overstocked_skus": inv.get("overstocked_skus"),
            })
            log_file_if_exists(inv_metrics_path, "metrics")
            log_file_if_exists(os.path.join(processed, "inventory_recommendations.parquet"), "data")
            log_file_if_exists(os.path.join(figures_dir, "inventory_health_analysis.png"), "figures")
            mlflow.set_tags({"project": "RetailPulse", "stage": "inventory"})
        print("   ✅ inventory run logged\n")

    # Summary
    exps = client.search_experiments()
    print("=" * 70)
    print("  LOGGED EXPERIMENTS")
    print("=" * 70)
    for e in exps:
        if e.name in {
            "Customer_Segmentation",
            "Demand_Forecasting",
            "Customer_Churn",
            "Inventory_Optimization",
        }:
            runs = client.search_runs([e.experiment_id], max_results=5)
            print(f"• {e.name}: {len(runs)} recent run(s)")

    print()
    print("🎉 MLflow tracking complete!")
    print()
    print("Next: open the MLflow UI with this command:")
    print(f'  python -m mlflow ui --backend-store-uri "{mlruns_dir}" --port 5000')
    print("Then open: http://127.0.0.1:5000")
    print("=" * 70)


if __name__ == "__main__":
    main()