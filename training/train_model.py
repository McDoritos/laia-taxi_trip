import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import pyarrow.parquet as pq
import os
import json
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 

from lightgbm import LGBMRegressor
# ============================================================
# CONFIGURAÇÕES INICIAIS
# ============================================================

COMMIT_SHA = os.getenv('COMMIT_SHA', 'local-dev')
if not COMMIT_SHA:
    raise EnvironmentError("Missing required env var: COMMIT_SHA")

MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'laia-taxi-model')
if not MODEL_NAME:
    raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

EXP_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'laia-taxi-exp')
if not EXP_NAME:
    raise EnvironmentError("Missing required env var: MLFLOW_EXPERIMENT_NAME")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://the-traffickers.dei.uc.pt:9002")
if not TRACKING_URI:
    raise EnvironmentError("Missing required env var: MLFLOW_TRACKING_URI")

# Read the variables
PATH_DATASET = os.getenv('PATH_DATASET',"../Dataset/")

# MLflow remoto (alterar IP conforme o servidor)
mlflow.set_tracking_uri(TRACKING_URI)

# Criar ou obter o experimento
mlflow.set_experiment(EXP_NAME)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def readDataset(root=None, sample_frac_per_file=0.05):
    """
    Read parquet files and return X, y for training a model compatible with the /predict payload.
    Only uses features present in the POST /predict payload.
    """
    use_cols = [
        "tpep_pickup_datetime",  # only needed for deriving time features
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "tpep_dropoff_datetime"  # needed to compute target y
    ]

    if root is None:
        root = os.environ.get("PATH_DATASET", "/app/Dataset/")

    pattern = os.path.join(root, "**", "yellow_tripdata_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    dfs = []
    for fpath in files:
        print(f"reading file: {fpath}")
        df = pd.read_parquet(
            fpath,
            engine="pyarrow",
            columns=[c for c in use_cols if c in pq.ParquetFile(fpath).schema.names],
        )
        if sample_frac_per_file and 0 < sample_frac_per_file < 1:
            df = df.sample(frac=sample_frac_per_file, random_state=123)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # compute target: duration in minutes
    pickup_col = "tpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime"
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 24 * 60)]

    # derived features from pickup timestamp
    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # location encoding
    df["PULocationID"] = df["PULocationID"].astype("category").cat.codes
    df["DOLocationID"] = df["DOLocationID"].astype("category").cat.codes


    feature_cols = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "PULocationID",
        "DOLocationID",
    ]

    X = df[feature_cols].fillna(0).reset_index(drop=True)
    y = df["duration_min"].values
    return X, y


def prediction_table(model, X, y_true, n=None, sort_by_error=False, ascending=False, save_csv=None):
    preds = model.predict(X)
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": preds,
        "error": preds - y_true,
        "abs_error": np.abs(preds - y_true)
    })
    if sort_by_error:
        df = df.sort_values("abs_error", ascending=ascending)
    if save_csv:
        df.to_csv(save_csv, index=False)
    return df.head(n) if n else df

# ============================================================
# training and tracking
# ============================================================

X, y = readDataset(sample_frac_per_file=0.001)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) # 42, 123

categorical_features = [
    c for c in [
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "PULocationID",
        "DOLocationID",
    ]
    if c in X_train.columns
]

# Inicia o run de MLflow
with mlflow.start_run(run_name="LightGBM_Training") as run:
    # -----------------------------
    # LightGBM fixed parameters
    # -----------------------------
    params = {
        "n_estimators": 500,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": -1,
        "random_state": 123,
        "verbose": -1,
    }

    mlflow.log_params(params)

    # -------------------------------------------------------------
    # ### ADDED: Save Reference Data for Drift Detection
    # -------------------------------------------------------------
    # We save the training features to a parquet file. 
    # This file will be downloaded by the monitoring system to compare against live traffic.
    ref_path = "reference.parquet"
    X_train.to_parquet(ref_path)
    
    print(f"Logging reference data to MLflow: {ref_path}")
    # We store it in a 'drift_info' folder inside the artifacts
    mlflow.log_artifact(ref_path, artifact_path="drift_info")
    # -------------------------------------------------------------
    
    model = LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )

   # -------------------------------------------------------------
    # ### Capture Baseline for Drift Detection (Evidently 0.7+)
    # -------------------------------------------------------------
    print("Generating training data baseline report...")

    # Explicit data definition (recommended)
    data_def = DataDefinition(
        numerical_columns=X_train.select_dtypes(include="number").columns.tolist(),
        categorical_columns=X_train.select_dtypes(exclude="number").columns.tolist()
    )

    # Build Evidently dataset
    train_dataset = Dataset.from_pandas(X_train, data_definition=data_def)

    # Create report
    report = Report(metrics=[DataDriftPreset()])

    # Run report (returns a Snapshot object)
    snapshot = report.run(
        reference_data=train_dataset,
        current_data=train_dataset
    )

    # Save JSON
    drift_report_path = "drift_baseline.json"
    snapshot.save_json(drift_report_path)
    # Log to MLflow
    mlflow.log_artifact(drift_report_path, artifact_path="drift_info")

    # -------------------------------------------------------------

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    print(f"Metrics:\n MAE={mae:.3f}, MSE={mse:.3f}, R2={r2:.3f}")

    # Guardar a tabela de predições
    pred_path = "predictions_sample.csv"
    prediction_table(model, X_test, y_test, n=50, sort_by_error=True, save_csv=pred_path)

    print(f"Saving predictions to mlflow {pred_path}...")
    mlflow.log_artifact(pred_path)

    # Infere model signature
    signature = mlflow.models.infer_signature(
                X_train, model.predict(X_train)
            )
    # Registrar modelo no MLflow Model Registry
    print("Registering model in MLflow Model Registry...")

    mlflow.sklearn.log_model(
        model, 
        name="model",
        signature=signature,
        input_example=X_train[:5]
        )
    
    model_uri = f"runs:/{run.info.run_id}/model"
    try:
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Model registered: {registered_model.name} (version {registered_model.version})")

        client = MlflowClient()
        # promote model to 'staging' and commit sha aliases
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="staging",
            version=registered_model.version
        )
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=COMMIT_SHA,
            version=registered_model.version,
        )

        print(f"Model version {registered_model.version} promoted to staging")

    except Exception as e:
        print(f"ERROR: Failed to register/promote model: {e}")
        print(f"Model URI: {model_uri}")
        print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        raise

print("\nExperiment finalized")