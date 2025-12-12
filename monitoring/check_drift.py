import pandas as pd
import mlflow
import json
import sys
import os
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
LOG_FILE = "logs/inference_logs.jsonl"

def check_drift():
    print("1. Loading Current Data (Inference Logs)...")
    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}. Skipping drift check.")
        return

    current_data = pd.read_json(LOG_FILE, lines=True)
    
    # Optional: Filter for recent data (e.g., last 24h)
    # current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
    # current_data = current_data[current_data['timestamp'] > pd.Timestamp.now() - pd.Timedelta(days=1)]

    if current_data.empty:
        print("Log file is empty. Skipping.")
        return

    print("2. Loading Reference Data (Training Set)...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    
    # Get the Production model version
    try:
        prod_model = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        run_id = prod_model.run_id
        print(f"Comparing against Production Model Version: {prod_model.version} (Run ID: {run_id})")
    except Exception:
        print("No production model found. Skipping.")
        return

    # Download reference data (Ensure you saved X_train as parquet in train_model.py!)
    # If you only saved the JSON report, Evidently requires the raw data for new comparisons.
    # Recommended: mlflow.log_artifact("reference.parquet") in training.
    local_ref_path = client.download_artifacts(run_id, "reference.parquet", dst_path=".")
    reference_data = pd.read_parquet(local_ref_path)

    # Align columns (Current data might have extra 'timestamp' or 'prediction' columns)
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    print(f"3. Running Drift Detection on {len(common_cols)} features...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save Report
    report.save_html("drift_report.html")
    print("Drift report saved to drift_report.html")

    # Parse Results
    results = report.as_dict()
    dataset_drift = results['metrics'][0]['result']['dataset_drift']
    
    if dataset_drift:
        print("!!! DATA DRIFT DETECTED !!!")

        # CHECK FOR DRY RUN MODE
        if os.environ.get("DRY_RUN") == "true":
            print(">>> DRY RUN: Retraining triggers are disabled.")
            print(">>> SUCCESS: The system correctly identified drift and attempted to retrain.")
            sys.exit(0) # Exit Green (Success)
        
        # 1. Define the GitHub API endpoint to trigger Stage 2
        # We trigger "2 - Continuous Delivery" (file: 2_continuous_delivery.yml)
        repo = "mcdoritos/laia-taxi_trip" # REPLACE WITH YOUR REPO
        workflow_id = "2_continuous_delivery.yml"
        token = os.environ.get("GITHUB_TOKEN") # Ensure this env var is set in your YAML
        
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/dispatches"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {"ref": "main"} # The branch to run on

        # 2. Trigger the Workflow
        print(f"Triggering retraining workflow: {workflow_id}...")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 204:
            print("Successfully triggered retraining.")
        else:
            print(f"Failed to trigger retraining: {response.status_code} - {response.text}")
            sys.exit(1) # Exit with error if we couldn't pull the alarm
            
        # We exit with 0 (Success) because the drift was handled successfully
        sys.exit(0)
    else:
        print("No drift detected.")

if __name__ == "__main__":
    check_drift()