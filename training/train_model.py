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


def readDataset(root=PATH_DATASET, sample_frac_per_file=0.05):
    use_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "pickup_datetime", "dropoff_datetime",
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "trip_distance", "PULocationID", "DOLocationID",
        "passenger_count", "fare_amount", "congestion_surcharge",
        "tip_amount", "total_amount"
    ]

    pattern = os.path.join(root, "**", "yellow_tripdata_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    sampled_dfs = []
    print("Starting to read parquet files...")
    for fpath in files:
        print(f"Reading {fpath}...")
        df = pd.read_parquet(
            fpath,
            engine="pyarrow",
            columns=[c for c in use_cols if c in pq.ParquetFile(fpath).schema.names],
        )
        if sample_frac_per_file and 0 < sample_frac_per_file < 1:
            df = df.sample(frac=sample_frac_per_file, random_state=123) # 42, 123
        sampled_dfs.append(df)

    df = pd.concat(sampled_dfs, ignore_index=True)

    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"

    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")

    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[df["duration_min"].notna()]
    df = df[(df["duration_min"] > 0.0) & (df["duration_min"] <= 24 * 60)]

    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["season"] = df["pickup_month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    if set(["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]).issubset(df.columns):
        df["haversine_km"] = haversine_vectorized(
            df["pickup_latitude"].astype(float).fillna(0.0),
            df["pickup_longitude"].astype(float).fillna(0.0),
            df["dropoff_latitude"].astype(float).fillna(0.0),
            df["dropoff_longitude"].astype(float).fillna(0.0),
        )
    else:
        df["haversine_km"] = df["trip_distance"].fillna(0.0)

    if "PULocationID" in df.columns:
        df["pu_zone_code"] = df["PULocationID"].astype("category").cat.codes
    if "DOLocationID" in df.columns:
        df["do_zone_code"] = df["DOLocationID"].astype("category").cat.codes

    df["has_congestion_fee"] = (df.get("congestion_surcharge", 0).fillna(0) > 0).astype(int)
    df["total_amount"] = df.get("total_amount", df.get("fare_amount", 0)).fillna(0)

    feature_cols = [
        "haversine_km", "trip_distance", "passenger_count", "fare_amount",
        "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
        "season", "is_rush_hour", "has_congestion_fee", "total_amount"
    ]
    if "pu_zone_code" in df.columns:
        feature_cols.append("pu_zone_code")
    if "do_zone_code" in df.columns:
        feature_cols.append("do_zone_code")

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
# TREINAMENTO E TRACKING
# ============================================================

X, y = readDataset(sample_frac_per_file=0.001)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) # 42, 123

# Inicia o run de MLflow
with mlflow.start_run(run_name="RandomForestRegressor_Training") as run:
    n_estimators = 100
    max_depth = 10
    min_samples_split = 10

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=123 # 42, 123
    )

    print("Training RandomForest...")
    model.fit(X_train, y_train)

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
        print(f"Modelo registrado: {registered_model.name} (versão {registered_model.version})")

        client = MlflowClient()
        # promote model to 'staging' and commit sha aliases
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=registered_model.version
        )
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=COMMIT_SHA,
            version=registered_model.version,
        )

        print(f"Model version {registered_model.version} promoted to Production")

    except Exception as e:
        print(f"ERROR: Failed to register/promote model: {e}")
        print(f"Model URI: {model_uri}")
        print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        raise

print("\nExperimento finalizado com sucesso!")
