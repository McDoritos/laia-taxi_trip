import pandas as pd
import mlflow
import json
import sys
import os
import requests
from evidently import Report
from evidently.presets import DataDriftPreset 

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LOG_FILE = "logs/inference_logs.jsonl"
HTML_REPORT_FILE = "drift_report.html"
JSON_REPORT_FILE = "drift_report.json"

def check_drift():
    # 1. LOAD CURRENT DATA
    print("1. Loading Current Data...")
    if not os.path.exists(LOG_FILE):
        print(f"File {LOG_FILE} not found.")
        sys.exit(0)

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
        print(f"Fetching model '{MLFLOW_MODEL_NAME}' with alias 'production'...")
        version_info = client.get_model_version_by_alias(name=MLFLOW_MODEL_NAME, alias="production")
        run_id = version_info.run_id
        
        local_ref_path = client.download_artifacts(run_id, "drift_info/reference.parquet", dst_path=".")
        reference_data = pd.read_parquet(local_ref_path)
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        sys.exit(1)

    # 3. ALIGN COLUMNS
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    exclude_cols = ["_prediction", "tpep_pickup_datetime"] 
    common_cols = [c for c in common_cols if c not in exclude_cols]

    if len(common_cols) == 0:
        print("ERROR: No overlapping columns.")
        sys.exit(1)

    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    # 4. RUN DRIFT REPORT
    print("4. Running Drift Report...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    snapshot.save_html(HTML_REPORT_FILE)
    snapshot.save_json(JSON_REPORT_FILE)

    # 5. PARSE DETAILED RESULTS
    # ✅ CORRECTED: Use .dict() based on your report.py file
    results = snapshot.dict()
    
    drift_detected = False
    drift_share = 0.0
    drift_count = 0
    
    
    metrics_list = results.get('metrics', [])
    
    found_drift_metric = False
    
    for metric in metrics_list:
        res = metric.get('result', {})
        
        # Check if this metric has the drift flags we need
        if isinstance(res, dict) and 'dataset_drift' in res:
            drift_detected = res['dataset_drift']
            drift_share = res.get('share_of_drifted_columns', 0.0)
            drift_count = res.get('number_of_drifted_columns', 0)
            found_drift_metric = True
            break
            
    if not found_drift_metric:
        print("WARNING: Could not find 'dataset_drift' in report results.")
        # Optional: Print keys to debug if it fails
        # print([m.get('metric') for m in metrics_list])

    # 6. LOG TO MLFLOW
    print("6. Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Weekly_Data_Drift_Checks")
    
    with mlflow.start_run(run_name="daily_check"):
        mlflow.log_artifact(HTML_REPORT_FILE)
        mlflow.log_artifact(JSON_REPORT_FILE)
        
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_metric("drift_count", drift_count)
        
        print(f"Drift Detected: {drift_detected}")
        print(f"Share of Drifted Cols: {drift_share}")

    # 7. TRIGGER RETRAINING
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
    resp = requests.post(url, headers=headers, json={"ref": "main"})
    
    if resp.status_code == 204:
        print("Workflow triggered successfully.")
    else:
        print(f"Failed to trigger workflow: {resp.text}")

if __name__ == "__main__":
    check_drift()