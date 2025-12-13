import os
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error

# Config
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
ALIAS_A = os.getenv("PROD_ALIAS", "production")
ALIAS_B = os.getenv("STAGING_ALIAS", "staging")
DATA_PATH = os.getenv("VALIDATION_DATA", "Dataset/2013.parquet")

# Load validation data
df = pd.read_parquet(DATA_PATH)
y_true = df["trip_duration"]
X = df.drop(columns=["trip_duration"])

# Derived features same as in app.py
X["tpep_pickup_datetime"] = pd.to_datetime(X["tpep_pickup_datetime"], errors="coerce")
X["pickup_hour"] = X["tpep_pickup_datetime"].dt.hour
X["pickup_dayofweek"] = X["tpep_pickup_datetime"].dt.weekday
X["pickup_month"] = X["tpep_pickup_datetime"].dt.month
X["is_weekend"] = X["pickup_dayofweek"].isin([5,6]).astype(int)
X["is_rush_hour"] = X["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

for col in ["PULocationID","DOLocationID"]:
    X[col] = X[col].astype("category").cat.codes
for col in ["pickup_hour","pickup_dayofweek","pickup_month","is_weekend","is_rush_hour","passenger_count","PULocationID","DOLocationID","VendorID"]:
    if col in X.columns:
        X[col] = X[col].astype(np.int32)
X["trip_distance"] = X["trip_distance"].astype(np.float64)

feature_cols = ["VendorID","trip_distance","passenger_count","pickup_hour","pickup_dayofweek","pickup_month","is_weekend","is_rush_hour","PULocationID","DOLocationID"]
X = X[feature_cols]

# Load models
client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))

model_a = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS_A}")
model_b = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS_B}")

# Predictions
y_pred_a = model_a.predict(X)
y_pred_b = model_b.predict(X)

# Metrics
rmse_a = mean_squared_error(y_true, y_pred_a, squared=False)
rmse_b = mean_squared_error(y_true, y_pred_b, squared=False)

print(f"Model {ALIAS_A}: RMSE={rmse_a:.3f}")
print(f"Model {ALIAS_B}: RMSE={rmse_b:.3f}")

if rmse_b < rmse_a:
    print(f"Model {ALIAS_B} performs better. Promote to production!")
    # Aqui podes chamar o teu código de promoção do MLflow
else:
    print(f"Model {ALIAS_A} remains in production.")
