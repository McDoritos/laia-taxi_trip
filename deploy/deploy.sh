#!/usr/bin/env bash
set -euo pipefail

# Load secrets
source .env

# -------------------------------
# Config (normally passed by CI)
# -------------------------------
REGISTRY="ghcr.io"
IMAGE_NAME="jrcampos/laia-tutorial/serving"
MLFLOW_TRACKING_URI="http://dsn2026hotcrp.dei.uc.pt:8080"
MLFLOW_MODEL_NAME="iris_cicd_model_joao_r_campos"
ALIAS="production"

# These two must be passed in environment before running script
: "${GITHUB_USERNAME:?Need GITHUB_USERNAME env var}"
: "${GITHUB_TOKEN:?Need GITHUB_TOKEN env var}"

# -------------------------------
# Authenticate to GHCR
# -------------------------------
echo "Logging into GHCR..."
echo "$GITHUB_TOKEN" | docker login "$REGISTRY" -u "$GITHUB_USERNAME" --password-stdin

# -------------------------------
# Pull image
# -------------------------------
echo "Pulling production image..."
docker pull "$REGISTRY/$IMAGE_NAME:$ALIAS"

# -------------------------------
# Stop existing container
# -------------------------------
if docker ps -q -f name=serving-app >/dev/null; then
    echo "Stopping existing container..."
    docker stop serving-app || true
    docker rm serving-app || true
else
    echo "No running container found, skipping stop."
fi

# -------------------------------
# Run new container
# -------------------------------
echo "Starting new serving-app container..."

docker run \
  --name serving-app \
  -p 9000:8080 \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e MLFLOW_MODEL_NAME="$MLFLOW_MODEL_NAME" \
  -e MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
  -e MLFLOW_TRACKING_PASSWORD="$MLFLOW_TRACKING_PASSWORD" \
  -e MODEL_ALIAS="$ALIAS" \
  "$REGISTRY/$IMAGE_NAME:production"

echo "✅ Deployment done successfully."