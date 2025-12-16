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

PATH_LIVE_DATA = os.getenv('PATH_DATASET', "data/retraining_dataset.parquet")
PATH_PAST_DATA = os.getenv('PATH_PAST_DATA', "Dataset/2013") 

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXP_NAME)

def traffic_period(hour: int) -> int:
    if 5 <= hour <= 7: return 0
    elif (9 <= hour <= 15) or (17 <= hour <= 18): return 2 
    else: return 1

def read_retraining(path, sample_frac=1.0):
    use_cols = [
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
        "duration_min"
    ]
    df = pd.read_parquet(path, columns=use_cols)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac)
        
    return df

def read_and_process_2013(root=None, sample_frac_per_file=0.05):
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
        root = os.environ.get("PATH_PAST_DATA", "/app/Dataset/2013")

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

    # Target Calcs
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
        "PULocationID",
        "DOLocationID",
        "traffic_period",
    ]

    return df

def get_X_Y(df):
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

def main():
    print("Starting retraining pipeline...")

    # 1. Load BOTH Datasets
    print("--- Loading Live Data ---")
    df_live = read_retraining(PATH_LIVE_DATA, sample_frac=1.0)
    print("--- Loading Past Data ---")
    df_past = read_and_process_2013(PATH_PAST_DATA, sample_frac_per_file=0.05)
    
    df_combined = pd.concat([df_past, df_live], ignore_index=True)

    # 2. Preprocess Combined Data
    X, y = get_X_Y(df_combined)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ["pickup_dayofweek", "pickup_month", "is_weekend", 
                            "is_rush_hour", "PULocationID", "DOLocationID", 
                            "VendorID", "traffic_period"]

    # 2. Start MLflow Run
    with mlflow.start_run(run_name=f"LightGBM_Retraining") as run:
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
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=COMMIT_SHA,
            version=registered_model.version
        )
        print(f"Model version {registered_model.version} promoted to alias 'staging'")

if __name__ == "__main__":
    main()