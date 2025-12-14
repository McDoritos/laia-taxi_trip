import os
import glob
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error
import pyarrow.parquet as pq

# --- Config ---
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
ALIAS_A = os.getenv("PROD_ALIAS", "production")
ALIAS_B = os.getenv("STAGING_ALIAS", "staging")
DATA_ROOT = os.getenv("VALIDATION_DATA", "Dataset/2013")  # folder, not a single file

# --- Helper to read dataset ---
def read_dataset(root=None, sample_frac_per_file=0.05):
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
        root = "/app/Dataset/2013"

    pattern = os.path.join(root, "**", "yellow_tripdata_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    dfs = []
    for fpath in files:
        print(f"Reading {fpath}")
        df = pd.read_parquet(
            fpath,
            engine="pyarrow",
            columns=[c for c in use_cols if c in pq.ParquetFile(fpath).schema.names],
        )
        if sample_frac_per_file and 0 < sample_frac_per_file < 1:
            df = df.sample(frac=sample_frac_per_file, random_state=123)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # compute target
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 24 * 60)]

    # derived features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.weekday
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    # location encoding
    df["PULocationID"] = df["PULocationID"].astype("category").cat.codes
    df["DOLocationID"] = df["DOLocationID"].astype("category").cat.codes

    feature_cols = [
        "VendorID",
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

# --- Load validation dataset ---
X_val, y_val = read_dataset(DATA_ROOT)

# --- Load models ---
client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
model_a = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS_A}")
model_b = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS_B}")

# --- Predictions ---
y_pred_a = model_a.predict(X_val)
y_pred_b = model_b.predict(X_val)

# --- Metrics ---
rmse_a = np.sqrt(mean_squared_error(y_val, y_pred_a))
rmse_b = np.sqrt(mean_squared_error(y_val, y_pred_b))

print(f"Model {ALIAS_A}: RMSE={rmse_a:.3f}")
print(f"Model {ALIAS_B}: RMSE={rmse_b:.3f}")

if rmse_b < rmse_a:
    print(f"Model {ALIAS_B} performs better. Promote to production!")
else:
    print(f"Model {ALIAS_A} remains in production.")
