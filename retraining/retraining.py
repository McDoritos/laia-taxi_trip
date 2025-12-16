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
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 

# --- Configuration ---
COMMIT_SHA = os.getenv('COMMIT_SHA', 'drift-retrain')
MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'laia-taxi-model')
EXP_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'laia-taxi-retraining')
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://the-traffickers.dei.uc.pt:9002")

# We now define TWO paths: one for live data, one for historical
PATH_LIVE_DATA = os.getenv('PATH_DATASET', "data/retraining_dataset.parquet")
PATH_PAST_DATA = os.getenv('PATH_PAST_DATA', "Dataset/2013") 

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXP_NAME)

def traffic_period(hour: int) -> int:
    if 5 <= hour <= 7: return 0
    elif (9 <= hour <= 15) or (17 <= hour <= 18): return 2 
    else: return 1

def load_parquet_data(path, sample_frac=1.0):
    """
    Helper to load parquet data from a single file or a directory of files.
    """
    dfs = []
    
    # Check if path is a file or directory
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.parquet"))
    else:
        print(f"Warning: Path not found {path}")
        return pd.DataFrame()

    if not files:
        print(f"Warning: No parquet files found in {path}")
        return pd.DataFrame()

    print(f"Loading data from {path} ({len(files)} files)...")
    
    for fpath in files:
        try:
            # We only read columns that actually exist in the file to avoid schema errors
            df = pd.read_parquet(fpath, engine="pyarrow")
            
            # Optional: Sample historical data if it's too huge
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
            
            dfs.append(df)
        except Exception as e:
            print(f"Skipping bad file {fpath}: {e}")

    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    """
    Applies all feature engineering and cleaning to a raw dataframe.
    """
    if df.empty:
        return df, None

    # --- 1. Label Normalization ---
    # Handle different names for the target variable
    if "duration_min" not in df.columns:
        if "true_duration" in df.columns:
            df["duration_min"] = df["true_duration"]
        elif "trip_time_in_secs" in df.columns: # Common in older NYC data
             df["duration_min"] = df["trip_time_in_secs"] / 60.0
        elif "tpep_dropoff_datetime" in df.columns and "tpep_pickup_datetime" in df.columns:
            df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
            df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
            df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    
    # Filter invalid targets
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 1440)] # Max 24h

    # --- 2. Feature Engineering ---
    # Create pickup_datetime if missing (some old datasets split date/time)
    if "tpep_pickup_datetime" not in df.columns:
         if "pickup_datetime" in df.columns:
             df["tpep_pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    if "pickup_hour" not in df.columns and "tpep_pickup_datetime" in df.columns:
         df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
         df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
         df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.weekday
         df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    # Ensure Derived Features exist
    if "traffic_period" not in df.columns and "pickup_hour" in df.columns:
        df["traffic_period"] = df["pickup_hour"].apply(traffic_period)
    
    if "is_weekend" not in df.columns and "pickup_dayofweek" in df.columns:
        df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

    if "is_rush_hour" not in df.columns and "pickup_hour" in df.columns:
        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # --- 3. Type Safety & Column Selection ---
    feature_cols = [
        "VendorID", "trip_distance", "passenger_count", 
        "pickup_hour", "pickup_dayofweek", "pickup_month", 
        "is_weekend", "is_rush_hour", "traffic_period", 
        "PULocationID", "DOLocationID"
    ]
    
    # Fill NAs and cast types
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Cast integers
    int_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend", 
                "is_rush_hour", "passenger_count", "traffic_period", 
                "PULocationID", "DOLocationID", "VendorID"]
    df[int_cols] = df[int_cols].astype(np.int32)
    
    return df[feature_cols], df["duration_min"]

def main():
    print("Starting retraining pipeline...")

    # 1. Load BOTH Datasets
    # Load Live Data (Keep 100% of it)
    print("--- Loading Live Data ---")
    df_live = load_parquet_data(PATH_LIVE_DATA, sample_frac=1.0)
    
    # Load Past Data (Sample 20% to avoid overwhelming the model with old data, adjust as needed)
    print("--- Loading Past Data ---")
    df_past = load_parquet_data(PATH_PAST_DATA, sample_frac=0.2) # Adjusted sample_frac

    # Combine them
    if df_live.empty and df_past.empty:
        raise ValueError("No data found in either Live or Past paths!")
    
    df_combined = pd.concat([df_past, df_live], ignore_index=True)
    print(f"Total Combined Rows: {len(df_combined)}")

    # 2. Preprocess Combined Data
    X, y = preprocess_data(df_combined)
    
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