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
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LOG_FILE = "logs/inference_logs.jsonl"

def check_drift():
    print("1. Loading Current Data (Inference Logs)...")
    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}. Skipping drift check.")
        return

    try:
        current_data = pd.read_json(LOG_FILE, lines=True)
    except ValueError:
        print("Log file format error or empty. Skipping.")
        return

    if current_data.empty:
        print("Log file is empty. Skipping.")
        return

    print("2. Loading Reference Data (Training Set)...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    
    try:
        prod_model = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        run_id = prod_model.run_id
        print(f"Comparing against Production Model Version: {prod_model.version} (Run ID: {run_id})")
        
        local_ref_path = client.download_artifacts(run_id, "reference.parquet", dst_path=".")
        reference_data = pd.read_parquet(local_ref_path)
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        return

    # Align columns
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    if not common_cols:
        print("Error: No common columns found.")
        sys.exit(1)

    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    print(f"3. Running Drift Detection on {len(common_cols)} features...")
    
    # --- FIXED: Use Snapshot API ---
    report = Report(metrics=[DataDriftPreset()])
    
    # run() returns a Snapshot object
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    # Save reports using the snapshot object
    snapshot.save_html("drift_report.html")
    snapshot.save_json("drift_report.json")
    print("Drift reports saved (html/json).")

    # Get the dictionary from the snapshot for logic checks
    results = snapshot.dict() 

    drift_detected = False
    
    # Parse metrics to find dataset_drift
    # Note: 'metric_types.py' uses 'value' to store results, not 'result'
    for metric in results.get('metrics', []):
        # We look for the 'DatasetDriftMetric' or similar output structure
        val = metric.get('value', {})
        
        # Check if 'dataset_drift' is in the value dict
        if isinstance(val, dict) and 'dataset_drift' in val:
            drift_detected = val['dataset_drift']
            break
            
    if drift_detected:
        print("!!! DATA DRIFT DETECTED !!!")

        if os.environ.get("DRY_RUN") == "true":
            print(">>> DRY RUN: Retraining triggers are disabled.")
            sys.exit(0) 
        
        print(f"Triggering retraining on {GITHUB_REPO}...")
        workflow_filename = "2_continuous_delivery.yml"
        
        if not GITHUB_REPO or not GITHUB_TOKEN:
            print("Error: GITHUB_REPOSITORY or GITHUB_TOKEN not set.")
            sys.exit(1)

        url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_filename}/dispatches"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {"ref": "main"}

        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 204:
            print("Successfully triggered retraining workflow.")
        else:
            print(f"Failed to trigger retraining: {response.status_code} - {response.text}")
            sys.exit(1)
    else:
        print("No drift detected. System stable.")

if __name__ == "__main__":
    check_drift()