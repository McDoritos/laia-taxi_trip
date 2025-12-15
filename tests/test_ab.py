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


def traffic_period(hour: int) -> int:
    """
    0 = low traffic (short trips)
    1 = medium traffic
    2 = high traffic (long trips)
    """
    if 5 <= hour <= 7:
        return 0  # low
    elif (9 <= hour <= 15) or (17 <= hour <= 18):
        return 2  # high
    else:
        return 1  # medium

# --- Helper to read dataset ---
def read_dataset(root=None, sample_frac_per_file=0.05):
    """
    Read parquet files and return X, y for training.
    ALIGNED with app.py /predict endpoint handling of NaNs and types.
    """
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
        root = os.environ.get("PATH_DATASET", "/app/Dataset/2013")

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

    # --- Target Calculation ---
    pickup_col = "tpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime"
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    
    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 24 * 60)]

    # --- Derived Features ---
    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["traffic_period"] = df["pickup_hour"].apply(traffic_period).astype(np.int32)

    
    # 1. ID Columns -> Fill NaN with -1
    id_cols = ["PULocationID", "DOLocationID", "VendorID"]
    for col in id_cols:
        # Coerce to numeric first (handles strings like "161"), then fillna(-1), then int32
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(np.int32)

    # 2. Count/Integer Columns -> Fill NaN with 0
    int_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month", 
                "is_weekend", "is_rush_hour", "passenger_count", "traffic_period"]
    for col in int_cols:
        df[col] = df[col].fillna(0).astype(np.int32)

    # 3. Float Columns -> Fill NaN with 0.0
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
        "PULocationID",
        "DOLocationID",
        "traffic_period",
    ]

    X = df[feature_cols].reset_index(drop=True)
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
