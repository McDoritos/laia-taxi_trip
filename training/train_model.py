import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from lightgbm import LGBMRegressor

import pandas as pd
import numpy as np
import os
import glob
import pyarrow.parquet as pq

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

if not all([COMMIT_SHA, MODEL_NAME, EXP_NAME, TRACKING_URI]):
    raise EnvironmentError("Missing critical environment variables")

PATH_DATASET = os.getenv('PATH_DATASET',"../Dataset")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXP_NAME)


# ============================================================
# LEITURA DE DADOS
# ============================================================

def readDataset(root=None, sample_frac_per_file=0.05):
    """
    Read parquet files and return X, y for training a model compatible with the /predict payload.
    Only uses features present in the POST /predict payload.
    """
    use_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "pickup_datetime", "dropoff_datetime",
        "trip_distance", "PULocationID", "DOLocationID",
        "passenger_count", "fare_amount", "congestion_surcharge",
        "total_amount"
    ]

    if root is None:
        root = PATH_DATASET

    pattern = os.path.join(root, "**", "yellow_tripdata_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    sampled_dfs = []

    print(f"Found {len(files)} parquet files, starting to read...")

    for fpath in files:
        file_cols = pq.ParquetFile(fpath).schema.names
        actual_cols = [c for c in use_cols if c in file_cols]
        
        print(f"Reading {fpath}...")
        
        df = pd.read_parquet(fpath, engine="pyarrow", columns=actual_cols)
        
        if sample_frac_per_file and 0 < sample_frac_per_file < 1:
            df = df.sample(frac=sample_frac_per_file, random_state=123)
        sampled_dfs.append(df)

    if not sampled_dfs:
        raise ValueError("Non data loaded.")

    df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")

    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"

    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[df["duration_min"].notna()]
    df = df[(df["duration_min"] > 1.0) & (df["duration_min"] <= 240.0)]
    df = df[df["trip_distance"] > 0.0]

    # derived features from pickup timestamp
    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["season"] = df["pickup_month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    if "PULocationID" in df.columns:
        df["pu_zone_code"] = df["PULocationID"].fillna(-1).astype("category").cat.codes
    if "DOLocationID" in df.columns:
        df["do_zone_code"] = df["DOLocationID"].fillna(-1).astype("category").cat.codes


    feature_cols = [
        "trip_distance", "passenger_count", "fare_amount",
        "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
        "season", "is_rush_hour", "has_congestion_fee", "total_amount",
        "pu_zone_code", "do_zone_code"
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).reset_index(drop=True)
    y = df["duration_min"].values
    return X, y, feature_cols

def prediction_table(model, X, y_true, n=None, sort_by_error=False, save_csv=None):
    preds = model.predict(X)
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": preds,
        "error": preds - y_true,
        "abs_error": np.abs(preds - y_true)
    })
    if sort_by_error:
        df = df.sort_values("abs_error", ascending=False)
    if save_csv:
        df.to_csv(save_csv, index=False)
    return df.head(n) if n else df

# ============================================================
# MAIN FLOW
# ============================================================

if __name__ == "__main__":
    print("Loading data")

    X, y, feature_names = readDataset(sample_frac_per_file=0.05)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Categorical features 
    categorical_features = ["pu_zone_code", "do_zone_code", "is_weekend", "is_rush_hour", "season"]
    categorical_features = [c for c in categorical_features if c in feature_names]

    with mlflow.start_run(run_name="LightGBM_Optimized") as run:
        
        print("Optimizing hyperparameters...")
        
        # Amostra pequena para tuning rápido
        X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=123)
        
        # Parâmetros específicos do LightGBM
        param_dist = {
            "n_estimators": [300, 500],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 50],
            "max_depth": [-1, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        lgbm = LGBMRegressor(n_jobs=-1, random_state=123, verbose=-1)

        search = RandomizedSearchCV(
            lgbm,
            param_distributions=param_dist,
            n_iter=8, 
            cv=2,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=-1
        )
        
        search.fit(X_tune, y_tune, categorical_feature=categorical_features)
        
        best_params = search.best_params_
        print(f"Best Parameters: {best_params}")
        mlflow.log_params(best_params)

        print("Training LightGMB...")
        
        model = LGBMRegressor(
            n_jobs=-1,
            random_state=123,
            verbose=-1,
            **best_params
        )
        
        model.fit(X_train, y_train, categorical_feature=categorical_features)

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
        mlflow.log_artifact(pred_path)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        print("Registering model in MLflow Model Registry...")
        
        mlflow.sklearn.log_model(
            model, 
            name="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        model_uri = f"runs:/{run.info.run_id}/model"
        
        try:
            registered_model = mlflow.register_model(model_uri, MODEL_NAME)
            client = MlflowClient()
            client.set_registered_model_alias(MODEL_NAME, "staging", registered_model.version)
            client.set_registered_model_alias(MODEL_NAME, COMMIT_SHA, registered_model.version)
            print(f"Model {registered_model.version} promoted to staging")
        except Exception as e:
            print(f"ERROR: Failed to register/promote model: {e}")
            print(f"Model URI: {model_uri}")
            print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
            raise

    print("Training Concluded.")