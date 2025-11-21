from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

# MOCKING FILE OF THE FLASK APP RUNNING ON THE REMOTE SERVER

app = Flask(__name__)

# For mock testing, start with MODEL as None
app.config["MODEL"] = None

MODEL_NAME = "dummy_model"
MODEL_ALIAS = "latest"


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)