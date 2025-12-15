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
    report = Report(metrics=[DataDriftPreset()],
                    include_tests=True)
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    snapshot.save_html(HTML_REPORT_FILE)
    snapshot.save_json(JSON_REPORT_FILE)

    # 5. PARSE DETAILED RESULTS
    results = snapshot.dict()
    metrics_list = results.get('metrics', [])
    drift_detected = False
    drift_share = 0.0
    drift_count = 0
    
    for metric in metrics_list:
        metric_config = metric.get('config', {})
        metric_type = metric_config.get('type', '')
        
        # Look specifically for the DriftedColumnsCount metric
        if metric_type == 'evidently:metric_v2:DriftedColumnsCount':
            val = metric.get('value', {})
            
            # Extract statistics
            drift_share = val.get('share', 0.0)
            drift_count = val.get('count', 0)
            
            # Calculate drift flag manually: share >= threshold (default 0.5)
            threshold = metric_config.get('drift_share', 0.5)
            drift_detected = drift_share >= threshold
            
            print(f"Drift Found: Share={drift_share:.3f}, Count={drift_count}, Threshold={threshold}")
            break

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
        sys.exit(10)
    else:
        print("System stable.")
        sys.exit(0)

if __name__ == "__main__":
    check_drift()