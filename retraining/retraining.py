import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import os
import glob
import pyarrow.parquet as pq
from evidently import Dataset, DataDefinition
from evidently.report import Report
from evidently.metric_preset import DataSummaryPreset

# --- Configuration ---
COMMIT_SHA = os.getenv('COMMIT_SHA', 'drift-retrain')
MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'laia-taxi-model')
EXP_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'laia-taxi-retraining')
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://the-traffickers.dei.uc.pt:9002")
PATH_DATASET = os.getenv('PATH_DATASET', "data/retraining_dataset.parquet")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXP_NAME)

def traffic_period(hour: int) -> int:
    if 5 <= hour <= 7: return 0
    elif (9 <= hour <= 15) or (17 <= hour <= 18): return 2 
    else: return 1

def read_flexible_dataset(path):
    """
    Robust loader that handles both:
    1. Raw Data (needs feature engineering)
    2. Log Data (already has features like pickup_hour)
    """
    print(f"Reading dataset from: {path}")
    
    # Handle directory vs single file
    if os.path.isfile(path):
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        files = glob.glob(os.path.join(path, "*.parquet"))
        if not files: raise FileNotFoundError(f"No parquet files in {path}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # --- 1. Label Check ---
    # In retraining, we might have 'duration_min' (calculated) or 'true_duration' (from logs)
    if "duration_min" not in df.columns:
        if "true_duration" in df.columns:
            df["duration_min"] = df["true_duration"]
        elif "tpep_dropoff_datetime" in df.columns and "tpep_pickup_datetime" in df.columns:
            print("Calculating duration from timestamps...")
            df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
            df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
            df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    
    # Filter invalid targets
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 1440)] # Max 24h

    # --- 2. Feature Engineering (Only if missing) ---
    if "pickup_hour" not in df.columns:
        print("Calculating time features...")
        if "tpep_pickup_datetime" in df.columns:
             df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
             df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
             df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.weekday
             df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
        else:
            raise ValueError("Cannot calculate features: 'tpep_pickup_datetime' missing.")

    # Ensure Derived Features exist
    if "traffic_period" not in df.columns:
        df["traffic_period"] = df["pickup_hour"].apply(traffic_period)
    
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

    if "is_rush_hour" not in df.columns:
        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # --- 3. Type Safety ---
    # Ensure all model input columns exist and are correct types
    feature_cols = [
        "VendorID", "trip_distance", "passenger_count", 
        "pickup_hour", "pickup_dayofweek", "pickup_month", 
        "is_weekend", "is_rush_hour", "traffic_period", 
        "PULocationID", "DOLocationID"
    ]
    
    # Fill NAs and cast types
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: Missing column {col}, filling with 0")
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Cast integers
    int_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend", 
                "is_rush_hour", "passenger_count", "traffic_period", 
                "PULocationID", "DOLocationID", "VendorID"]
    df[int_cols] = df[int_cols].astype(np.int32)
    
    return df[feature_cols], df["duration_min"]

def main():
    # 1. Load Data
    # NO SAMPLING: For retraining, we want to use all the hard-earned live data we have.
    X, y = read_flexible_dataset(PATH_DATASET)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ["pickup_dayofweek", "pickup_month", "is_weekend", 
                            "is_rush_hour", "PULocationID", "DOLocationID", 
                            "VendorID", "traffic_period"]

    # 2. Start MLflow Run
    with mlflow.start_run(run_name=f"Retrain_{COMMIT_SHA}") as run:
        params = {
            "n_estimators": 800,
            "learning_rate": 0.1,
            "num_leaves": 70,
            "max_depth": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # 3. Train
        print("Training model...")
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, categorical_feature=categorical_features)

        # 4. Evaluate
        preds = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        }
        mlflow.log_metrics(metrics)
        print(f"Retraining Metrics: {metrics}")

        # 5. Register Model
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        
        # 6. Promote to Staging (Candidate for Production)
        client = MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="staging",
            version=registered_model.version
        )
        print(f"Model version {registered_model.version} promoted to alias 'staging'")

if __name__ == "__main__":
    main()