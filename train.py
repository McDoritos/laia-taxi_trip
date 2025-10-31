import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


PATH_DATASET = r"F:\Universidade\LAIA\laia-taxi_trip\Dataset"

# Configurar o MLflow para usar o servidor remoto com upload de artefatos
#mlflow.set_tracking_uri("http://localhost:5000")
#
## Nome do experimento e URI remoto para artefatos
#experiment_name = "taxi_classification"
#artifact_uri = "mlflow-artifacts:/"
#
## Verificar se o experimento já existe e tem o local de artefatos correto
#existing_experiment = mlflow.get_experiment_by_name(experiment_name)
#
#if existing_experiment:
#    # Verifica se o experimento existente tem um caminho problemático de artefatos (ex: Docker local)
#    if existing_experiment.artifact_location.startswith("/mlflow") or \
#       existing_experiment.artifact_location.startswith("file:///mlflow"):
#        print(f"Existing experiment has Docker path artifact location: {existing_experiment.artifact_location}")
#        print("Creating new experiment with remote artifact storage...")
#
#        # Usa um novo nome de experimento
#        experiment_name = "taxi_classification_remote"
#        try:
#            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
#            print(f"Created new experiment '{experiment_name}'")
#        except Exception:
#            pass
#    else:
#        print(f"Using existing experiment '{experiment_name}'")
#else:
#    # Criar novo experimento com local de artefatos remoto
#    try:
#        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
#        print(f"Created new experiment '{experiment_name}' with remote artifact storage")
#    except Exception as e:
#        print(f"Note: {e}")
#
#mlflow.set_experiment(experiment_name)

def haversine_vectorized(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
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
        df = pd.read_parquet(fpath, engine="pyarrow", columns=[c for c in use_cols if c in pq.ParquetFile(fpath).schema.names])
        if sample_frac_per_file and 0 < sample_frac_per_file < 1:
            df = df.sample(frac=sample_frac_per_file, random_state=42)
        sampled_dfs.append(df)

    df = pd.concat(sampled_dfs, ignore_index=True)

    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"

    print("Trip duration: Calculating...")
    # parse datetimes
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")

    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0

    print("Trip duration: removing bad rows...")
    # filter obvious bad rows
    df = df[df["duration_min"].notna()]
    df = df[(df["duration_min"] > 0.0) & (df["duration_min"] <= 24*60)]

    print("Temporal features: extracting...")
    # temporal features
    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5,6]).astype(int)
    
    print("Spatial features: calculating...")
    # season (simple mapping)
    df["season"] = df["pickup_month"].map({12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}).astype(int)
    
    print("Rush hour feature: creating...")
    # rush hour flag
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    print("Haversine distance: calculating...")
    # spatial features: haversine distance
    if set(["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]).issubset(df.columns):
        df["haversine_km"] = haversine_vectorized(
            df["pickup_latitude"].astype(float).fillna(0.0),
            df["pickup_longitude"].astype(float).fillna(0.0),
            df["dropoff_latitude"].astype(float).fillna(0.0),
            df["dropoff_longitude"].astype(float).fillna(0.0),
        )
    else:
        df["haversine_km"] = df["trip_distance"].fillna(0.0)  # fallback

    print("Zone ID features: encoding...")
    # zone IDs as categorical codes (compact numeric)
    if "PULocationID" in df.columns:
        df["pu_zone_code"] = df["PULocationID"].astype("category").cat.codes
    if "DOLocationID" in df.columns:
        df["do_zone_code"] = df["DOLocationID"].astype("category").cat.codes

    print("Contextual features: creating...")
    # contextual features
    df["has_congestion_fee"] = (df.get("congestion_surcharge", 0).fillna(0) > 0).astype(int)
    df["total_amount"] = df.get("total_amount", df.get("fare_amount", 0)).fillna(0)

    print("Assembling feature matrix...")
    # assemble feature matrix (choose/extend as needed)
    feature_cols = [
        "haversine_km", "trip_distance", "passenger_count", "fare_amount",
        "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
        "season", "is_rush_hour", "has_congestion_fee", "total_amount"
    ]
    # include zone codes if present
    if "pu_zone_code" in df.columns:
        feature_cols.append("pu_zone_code")
    if "do_zone_code" in df.columns:
        feature_cols.append("do_zone_code")

    X = df[feature_cols].fillna(0).reset_index(drop=True)
    y = df["duration_min"].values

    print(f"Loaded {len(df)} rows (sample_frac_per_file={sample_frac_per_file}).")
    return X, y

def print_parquet_columns(root=PATH_DATASET, max_files=None):
    pattern = os.path.join(root, "**", "yellow_tripdata_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    unique_cols = set()
    for i, fpath in enumerate(files):
        pf = pq.ParquetFile(fpath)
        cols = pf.schema.names
        unique_cols.update(cols)
        if max_files and (i + 1) >= max_files:
            break

    print(sorted(unique_cols))


X, y = readDataset(sample_frac_per_file=0.01)   # sample 1% from each file
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_rf_search(X_train, y_train, X_val, y_val, n_iter=20):
    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [8, 16, None],
        "min_samples_split": [2, 5, 10]
    }
    rs = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, scoring="neg_mean_absolute_error", cv=3, n_jobs=4, random_state=42, verbose=3)
    rs.fit(X_train, y_train)
    best = rs.best_estimator_
    preds = best.predict(X_val)
    print("RF MAE:", mean_absolute_error(y_val, preds))
    return best

def prediction_table(model, X, y_true, n=None, sort_by_error=False, ascending=False, save_csv=None, round_decimals=3):
    preds = model.predict(X)

    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.round(preds, round_decimals)
    })
    df["error"] = np.round(df["y_pred"] - df["y_true"], round_decimals)
    df["abs_error"] = np.round(df["error"].abs(), round_decimals)

    if sort_by_error:
        df = df.sort_values("abs_error", ascending=ascending).reset_index(drop=True)

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"Saved predictions to {save_csv}")

    return df.head(n) if isinstance(n, int) else df


#If there isn't a model registered yet, train and register the model
#model = train_rf_search(X_train, y_train, X_test, y_test)

#If there is already a model registered, just create a new model instance with the same hyperparameters, ideally it should be loaded from the registry
model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=10, n_jobs=-1, random_state=42)
print("Training RandomForest...")
model.fit(X_train, y_train)
print("Evaluating RandomForest...")
preds = model.predict(X_test)
print("RF MAE:", mean_absolute_error(y_test, preds))
print("RF MSE:", root_mean_squared_error(y_test, preds))
print("RF R2:", r2_score(y_test, preds))
print("\nSaving results")
prediction_table(model, X_test, y_test, n=50, sort_by_error=True, save_csv="predictions_sample.csv")

## Registrar o melhor modelo no MLflow Model Registry
#print(f"\nBest model: C={best_C}, accuracy={best_acc:.4f}")
#model_uri = f"runs:/{best_run_id}/model"
#registered_model = mlflow.register_model(model_uri, "iris")
#print(f"Registered model version: {registered_model.version}")
#