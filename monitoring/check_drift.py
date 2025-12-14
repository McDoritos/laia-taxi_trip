import pandas as pd
import mlflow
import json
import sys
import os
import requests
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LOG_FILE = "logs/inference_logs.jsonl"
DRIFT_REPORT_FILE = "drift_report.html"
REPORT_FILE = "drift_report.html"

def check_drift():
    # 1. LOAD CURRENT DATA
    print("1. Loading Current Data...")
    if not os.path.exists(LOG_FILE):
        print(f"File {LOG_FILE} not found.")
        sys.exit(0) # Exit gracefully if no logs

    try:
        current_data = pd.read_json(LOG_FILE, lines=True)
    except ValueError as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)

    # 2. LOAD REFERENCE DATA
    print("2. Loading Reference Data...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    
    try:
        # Fetch the production model version
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        # Or use alias if preferred:
        # version_info = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        
        if not versions:
            print("No production model found.")
            sys.exit(1)
            
        run_id = versions[0].run_id
        print(f"Reference Run ID: {run_id}")
        
        local_ref_path = client.download_artifacts(run_id, "drift_info/reference.parquet", dst_path=".")
        reference_data = pd.read_parquet(local_ref_path)
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        sys.exit(1)

    # 3. ALIGN COLUMNS (CRITICAL STEP)
    # We only want to compare features that exist in both
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    
    # Filter out non-feature columns if they leaked in (like 'prediction' or IDs if not needed)
    exclude_cols = ["_prediction", "tpep_pickup_datetime"] 
    common_cols = [c for c in common_cols if c not in exclude_cols]

    if len(common_cols) == 0:
        print("ERROR: No overlapping columns between reference and current data.")
        print(f"Reference cols: {reference_data.columns.tolist()}")
        print(f"Current cols: {current_data.columns.tolist()}")
        sys.exit(1)

    print(f"Comparing columns: {common_cols}")
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    # 4. RUN DRIFT (Same as before)
    print("4. Running Drift Report...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    # Save locally first (required for upload)
    snapshot.save_html(REPORT_FILE)
    
    # Check for drift result (Logic from your original script)
    results = snapshot.as_dict() if hasattr(snapshot, "as_dict") else snapshot.dict()
    drift_detected = False
    
    # Parse metric results
    metrics_list = results.get('metrics', [])
    for metric in metrics_list:
        val = metric.get('result', {}) or metric.get('value', {})
        if isinstance(val, dict) and 'dataset_drift' in val:
            drift_detected = val['dataset_drift']
            break

    # ==========================================
    # 5. UPLOAD TO MLFLOW (NEW STEP)
    # ==========================================
    print("5. Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Use a specific experiment for monitoring so it doesn't mix with training runs
    mlflow.set_experiment("Weekly_Data_Drift_Checks")
    
    with mlflow.start_run(run_name="daily_check"):
        # Log the HTML report so you can download/view it in UI
        mlflow.log_artifact(REPORT_FILE)
        
        # Log the boolean result (0 or 1) so you can plot it over time
        mlflow.log_metric("drift_detected", int(drift_detected))
        
        # Optional: Log the share of drifting features
        # mlflow.log_metric("drift_share", ...) 

    print("Report uploaded to MLflow.")

    # 6. TRIGGER RETRAINING (Same as before)
    if drift_detected:
        print("!!! DATA DRIFT DETECTED !!!")
        trigger_retraining()
    else:
        print("System stable.")

def trigger_retraining():
    if os.environ.get("DRY_RUN") == "true":
        print("DRY RUN: Skipping Github Action Trigger.")
        return

    print("Triggering retraining workflow...")
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/2_continuous_delivery.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    # dispatch event requires 'ref'
    resp = requests.post(url, headers=headers, json={"ref": "main"})
    
    if resp.status_code == 204:
        print("Workflow triggered successfully.")
    else:
        print(f"Failed to trigger workflow: {resp.text}")

if __name__ == "__main__":
    check_drift()