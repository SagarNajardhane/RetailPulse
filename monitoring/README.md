# RetailPulse Monitoring & MLOps

## MLflow Tracking

### Log saved project metrics/models
```bash
python monitoring/mlflow_tracking.py

python monitoring/mlflow_churn_live_run.py


python -m mlflow ui --backend-store-uri "./mlruns" --port 5000