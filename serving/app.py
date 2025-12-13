from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
import json

LOG_FILE = "logs/inference_logs.jsonl"

def log_inference(df: pd.DataFrame, predictions):
    """Log each inference request and prediction to a JSONL file."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    admin_uid = int(os.getenv("ADMIN_UID", os.getuid()))
    admin_gid = int(os.getenv("ADMIN_GID", os.getgid()))
    os.chown(os.path.dirname(LOG_FILE), admin_uid, admin_gid)

    records = df.to_dict(orient="records")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for row, pred in zip(records, predictions.tolist()):
            row['_prediction'] = pred
            f.write(json.dumps(row) + "\n")

    os.chown(LOG_FILE, admin_uid, admin_gid)
    os.chmod(LOG_FILE, 0o666)

os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"
mlflow.set_tracking_uri("http://the-traffickers-internal.dei.uc.pt:5050")
app = Flask(__name__)

MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
MODEL_ALIAS = os.getenv("MODEL_ALIAS")

try:
    app.config["MODEL"] = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )
    print("Model loaded successfully at startup.")
except Exception as e:
    app.config["MODEL"] = None
    print(f"Could not load model at startup: {e}")

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return current model alias version + run ID hash"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return jsonify(
            model_name=MODEL_NAME,
            alias=MODEL_ALIAS,
            version=alias_info.version,
            run_id=alias_info.run_id,
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="healthy", model_loaded=app.config["MODEL"] is not None)

@app.route('/predict', methods=['POST'])
def predict():
    model = app.config.get("MODEL")
    if model is None:
        return jsonify({"error": "Model is not loaded. Check /health or logs."}), 503

    json_input = request.get_json()
    if 'data' not in json_input:
        return jsonify({"error": "Missing 'data' in request"}), 400

    df = pd.DataFrame(json_input['data'], columns=json_input['columns'])

    # --- DERIVED FEATURES ---
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.weekday
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5,6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    # Encode categorical features
    df["PULocationID"] = df["PULocationID"].astype("category").cat.codes
    df["DOLocationID"] = df["DOLocationID"].astype("category").cat.codes

    # Cast numeric types
    int_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month",
                "is_weekend", "is_rush_hour", "passenger_count",
                "PULocationID", "DOLocationID", "VendorID"]
    float_cols = ["trip_distance"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int32)
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)

    # Features for the model
    feature_cols = [
        "VendorID", "trip_distance", "passenger_count",
        "pickup_hour", "pickup_dayofweek", "pickup_month",
        "is_weekend", "is_rush_hour", "PULocationID", "DOLocationID"
    ]
    X = df[feature_cols]

    predictions = model.predict(X)
    log_inference(X, predictions)

    return jsonify(predictions.tolist())

@app.route("/reload", methods=["GET"])
def reload_model():
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        app.config["MODEL"] = model
        return jsonify(message="Model reloaded successfully.")
    except Exception as e:
        return jsonify(error=f"Failed to load model: {e}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
