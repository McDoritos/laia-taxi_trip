from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

# Permitir que todos os hosts se conectem ao MLflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configurar o URI de rastreamento do MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

app = Flask(__name__)

MODEL_NAME = "iris"
MODEL_STAGE = "Production"

# Tentar carregar o modelo na inicialização
try:
    app.config["MODEL"] = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    print("✅ Model loaded successfully at startup.")
except Exception as e:
    app.config["MODEL"] = None
    print(f"⚠️ Could not load model at startup: {e}")
    print("App will start without a model. You can load it later using /reload.")

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

    data = request.get_json()
    df = pd.DataFrame(data["data"], columns=data["columns"])
    preds = model.predict(df)
    return jsonify(predictions=preds.tolist())

@app.route("/reload", methods=["POST"])
def reload_model():
    """Reload model from MLflow and store in Flask app config."""
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        app.config["MODEL"] = model
        return jsonify(message="Model reloaded successfully.")
    except Exception as e:
        return jsonify(error=f"Failed to load model: {e}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
