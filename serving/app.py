from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

# Allow all hosts to connect to Mlflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configure MLflow tracking URI and authentication
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
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


@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the model stored in app config."""
    model = app.config["MODEL"]
    if model is None:
        return jsonify(error="Model not loaded. Please call /reload first."), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify(error="No JSON data provided"), 400
        
        if "data" not in data or "columns" not in data:
            return jsonify(error="Missing required fields: 'data' and 'columns'"), 400
        
        df = pd.DataFrame(data["data"], columns=data["columns"])
        preds = model.predict(df)
        return jsonify(predictions=preds.tolist())
    except KeyError as e:
        return jsonify(error=f"Missing required field: {str(e)}"), 400
    except ValueError as e:
        return jsonify(error=f"Invalid data format: {str(e)}"), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {str(e)}"), 500


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