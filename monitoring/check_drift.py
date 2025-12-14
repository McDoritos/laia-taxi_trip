import pandas as pd
import mlflow
import json
import sys
import os
import requests
from evidently import Report
from evidently.presets import DataDriftPreset 

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
    
    # Save locally
    snapshot.save_html(HTML_REPORT_FILE)
    snapshot.save_json(JSON_REPORT_FILE)

    # 5. PARSE DETAILED RESULTS
    # evidently's as_dict() returns a structure we can search
    results = snapshot.as_dict()
    
    drift_detected = False
    drift_share = 0.0
    drift_count = 0
    
    # Locate the DatasetDriftMetric inside the report
    metrics_list = results.get('metrics', [])
    for metric in metrics_list:
        if metric['metric'] == 'DatasetDriftMetric':
            result_val = metric.get('result', {})
            drift_detected = result_val.get('dataset_drift', False)
            drift_share = result_val.get('share_of_drifted_columns', 0.0)
            drift_count = result_val.get('number_of_drifted_columns', 0)
            break

    # 6. LOG TO MLFLOW
    print("6. Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Weekly_Data_Drift_Checks")
    
    with mlflow.start_run(run_name="daily_check"):
        # Log Artifacts (Visuals & Raw Data)
        mlflow.log_artifact(HTML_REPORT_FILE)
        mlflow.log_artifact(JSON_REPORT_FILE)
        
        # Log Metrics (Graphable Numbers)
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_share", drift_share)  # e.g., 0.15 (15%)
        mlflow.log_metric("drift_count", drift_count)  # e.g., 3 columns
        
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