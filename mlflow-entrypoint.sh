#!/bin/bash
export MLFLOW_ARTIFACTS_DESTINATION=/mlflow/mlruns
# Start MLflow server with all hosts allowed
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --serve-artifacts \
    --allowed-hosts '*'