from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
import json
import pwd
import grp

LOG_FILE = "logs/inference_logs.jsonl"

def log_inference(df: pd.DataFrame, predictions):
    """
    Log each inference request and prediction to a JSONL file
    for later data drift detection.
    
    Args:
        df: DataFrame containing input features for prediction.
        predictions: List or array of predictions.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    records = df.to_dict(orient="records")

    for row, pred in zip(records, predictions.tolist()):
        row['_prediction'] = pred
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Corrigir permissões
    os.chmod(LOG_FILE, 0o664)  # rw-rw-r--

    try:
        # Forçar dono admin (usuário dentro do container)
        uid = pwd.getpwnam("admin").pw_uid
        gid = grp.getgrnam("admin").gr_gid
        os.chown(LOG_FILE, uid, gid)
    except KeyError:
        # Se usuário "admin" não existir, apenas ignora
        pass

# Allow all hosts to connect to Mlflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configure MLflow tracking URI and authentication
mlflow.set_tracking_uri("http://the-traffickers-internal.dei.uc.pt:5050")
app = Flask(__name__)

MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
if not MODEL_NAME:
    raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

MODEL_ALIAS = os.getenv("MODEL_ALIAS")

# Try to load model once on startup
try:
    app.config["MODEL"] = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )
    print("Model loaded successfully at startup.")
except Exception as e:
    app.config["MODEL"] = None
    print(f"Could not load model at startup: {e}")
    print("App will start without a model. You can load it later using /reload.")


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
            run_id=alias_info.run_id,  # Commit SHA is stored in run_id (if you passed it)
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify(status="healthy", model_loaded=app.config["MODEL"] is not None)


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Retrieve the model from the app config
    model = app.config.get("MODEL")
    
    # Safety check: ensure model exists
    if model is None:
        return jsonify({"error": "Model is not loaded. Check /health or logs."}), 503

    json_input = request.get_json()

     # 2. Create DataFrame with model's expected columns
    required_columns = [
        "haversine_km", "trip_distance", "passenger_count", "fare_amount",
        "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
        "season", "is_rush_hour", "has_congestion_fee", "total_amount",
        "pu_zone_code", "do_zone_code"
    ]

    # Check if all required columns are present
    if 'columns' not in json_input or 'data' not in json_input:
        return jsonify({"error": "Missing 'data' or 'columns' in request"}), 400
    
    # 2. Create DataFrame from JSON
    df = pd.DataFrame(json_input['data'], columns=json_input['columns'])

    # 3. Cast integer columns
    int_cols = ['pickup_hour', 'pickup_dayofweek', 'pickup_month',
                'pu_zone_code', 'do_zone_code', 'passenger_count',
                'is_weekend', 'season', 'is_rush_hour', 'has_congestion_fee']

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int32)

        # 4. Cast float columns
    float_cols = ['haversine_km', 'trip_distance', 'fare_amount', 'total_amount']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)

    # 5. Make Prediction
    predictions = model.predict(df)

    # --- NEW STEP: Log the data for monitoring ---
    # We run this AFTER prediction so we can log the result too
    # Log for monitoring
    log_inference(df, predictions)

    # Return predictions
    return jsonify(predictions.tolist())

@app.route("/reload", methods=["GET"])
def reload_model():
    """Reload model from MLflow and store in Flask app config."""
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        app.config["MODEL"] = model
        return jsonify(message="Model reloaded successfully.")
    except Exception as e:
        return jsonify(error=f"Failed to load model: {e}"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)