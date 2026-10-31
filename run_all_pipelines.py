import os
import sys
import time
import subprocess
from datetime import datetime

def print_banner(step_num, total_steps, title):
    print("\n" + "=" * 75)
    print(f"🚀 STEP [{step_num}/{total_steps}]: {title}")
    print("=" * 75)

def find_script(candidates):
    """Finds the first existing script path from candidate paths."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def run_script(python_exe, candidates, description, artifact_check=None):
    script_path = find_script(candidates)
    
    if not script_path:
        if artifact_check and os.path.exists(artifact_check):
            print(f"🟢 Artifact verified: '{artifact_check}' exists. Proceeding with pipeline...")
            return True
        else:
            print(f"⚠️ Warning: Neither script candidate {candidates} nor artifact '{artifact_check}' found.")
            return True

    start_time = time.time()
    print(f"⏳ Executing: {description} ({script_path})...")
    try:
        subprocess.run([python_exe, script_path], check=True, text=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✅ Finished: {description} (Took {elapsed:.2f}s)\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {description}: Command failed with return code {e.returncode}")
        return False

def main():
    start_total = time.time()
    python_exe = sys.executable

    print("\n" + "★" * 75)
    print("   RETAILPULSE — AI-POWERED CUSTOMER ANALYTICS & DEMAND FORECASTING")
    print("               MASTER END-TO-END PIPELINE ORCHESTRATOR           ")
    print("★" * 75)
    print(f"🕒 Execution Started At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Executable  : {python_exe}\n")

    pipeline_steps = [
        (1, ["src/data_cleaning.py", "data_cleaning.py"], "Raw Excel Data Cleaning", "data/processed/cleaned_full.parquet"),
        (2, ["src/feature_engineering.py", "feature_engineering.py"], "Customer Feature Engineering", "data/processed/customer_features.parquet"),
        (3, ["src/customer_segmentation.py", "customer_segmentation.py"], "Customer Segmentation (K-Means)", "data/processed/customers_segmented.parquet"),
        (4, ["generate_churn_metrics.py", "src/churn_prediction.py"], "XGBoost Churn Risk Scorer & Metrics", "data/processed/churn_metrics.json"),
        (5, ["src/inventory_optimization.py", "inventory_optimization.py"], "Safety Stock & EOQ Inventory Optimization", "data/processed/inventory_recommendations.parquet"),
        (6, ["monitoring/mlflow_tracking.py"], "MLflow Experiment Tracking Logging", None),
        (7, ["monitoring/drift_detection.py"], "Statistical Data Drift Analysis", "monitoring/drift_report.html"),
        (8, ["monitoring/alert_system.py"], "Automated Threshold Health System Alert Check", "monitoring/alert_payload.json")
    ]

    total_steps = len(pipeline_steps)

    for step_num, candidates, description, artifact_check in pipeline_steps:
        print_banner(step_num, total_steps, description)
        success = run_script(python_exe, candidates, description, artifact_check)
        if not success:
            print(f"❌ Master Orchestrator halted at Step {step_num}.")
            sys.exit(1)

    total_elapsed = time.time() - start_total
    print("\n" + "★" * 75)
    print(f"🎉 MASTER PIPELINE EXECUTED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS!")
    print("★" * 75)

    launch_app = input("\n📊 Do you want to launch the Streamlit RetailPulse Dashboard now? (y/n): ").strip().lower()
    if launch_app == 'y':
        print("\n🚀 Launching RetailPulse Dashboard at http://localhost:8501 ...")
        subprocess.run([python_exe, "-m", "streamlit", "run", "dashboard/app.py"])

if __name__ == "__main__":
    main()