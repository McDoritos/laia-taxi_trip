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

PATH_DATASET = os.getenv('PATH_DATASET',"../Dataset/")

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXP_NAME)


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def traffic_period(hour: int) -> int:
    """
    0 = low traffic
    1 = medium traffic
    2 = high traffic 
    """
    if 5 <= hour <= 7:
        return 0
    elif (9 <= hour <= 15) or (17 <= hour <= 18):
        return 2 
    else:
        return 1

    """ Read parquet files and return X, y for training."""
def readDataset(root=None, sample_frac_per_file=0.05):
    use_cols = [
        "tpep_pickup_datetime", 
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "tpep_dropoff_datetime",
        "VendorID"
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

    pickup_col = "tpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime"
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    
    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 24 * 60)]

    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["traffic_period"] = df["pickup_hour"].apply(traffic_period).astype(np.int32)

    
    id_cols = ["PULocationID", "DOLocationID", "VendorID"]
    for col in id_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(np.int32)

    int_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month", 
                "is_weekend", "is_rush_hour", "passenger_count", "traffic_period"]
    for col in int_cols:
        df[col] = df[col].fillna(0).astype(np.int32)

    float_cols = ["trip_distance"]
    for col in float_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)

    feature_cols = [
        "VendorID",
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "traffic_period",
        "PULocationID",
        "DOLocationID",
    ]

    X = df[feature_cols].reset_index(drop=True)
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
        "VendorID",
        "traffic_period"
    ]
    if c in X_train.columns
]


with mlflow.start_run(run_name="LightGBM_Training") as run:
    params = {
        "n_estimators": 800,
        "learning_rate": 0.1,
        "num_leaves": 70,
        "max_depth": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 123,
        "verbose": -1,
    }

    mlflow.log_params(params)
    ref_path = "reference.parquet"
    X_train.to_parquet(ref_path)
    
    print(f"Logging reference data to MLflow: {ref_path}")
    mlflow.log_artifact(ref_path, artifact_path="drift_info")
  
    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )

    print("Generating training data baseline report...")

    data_def = DataDefinition(
        numerical_columns=X_train.select_dtypes(include="number").columns.tolist(),
        categorical_columns=X_train.select_dtypes(exclude="number").columns.tolist()
    )

    train_dataset = Dataset.from_pandas(X_train, data_definition=data_def)
    report = Report(metrics=[DataSummaryPreset()])

    snapshot = report.run(
        reference_data=None, 
        current_data=train_dataset
    )

    summary_report_path = "training_data_summary.json"
    snapshot.save_json(summary_report_path)
    mlflow.log_artifact(summary_report_path, artifact_path="drift_info")

    html_path = "training_data_summary.html"
    snapshot.save_html(html_path)
    mlflow.log_artifact(html_path, artifact_path="drift_info")



    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    print(f"Metrics:\n MAE={mae:.3f}, MSE={mse:.3f}, R2={r2:.3f}")

 
    pred_path = "predictions_sample.csv"
    prediction_table(model, X_test, y_test, n=50, sort_by_error=True, save_csv=pred_path)

    print(f"Saving predictions to mlflow {pred_path}...")
    mlflow.log_artifact(pred_path)

    signature = mlflow.models.infer_signature(
                X_train, model.predict(X_train)
            )
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