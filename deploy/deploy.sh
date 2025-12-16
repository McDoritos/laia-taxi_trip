#!/usr/bin/env bash
set -euo pipefail


if [ "$#" -ne 4 ]; then
    echo "Usage: $0 GITHUB_USERNAME GITHUB_TOKEN MLFLOW_TRACKING_URI MLFLOW_MODEL_NAME"
    exit 1
fi

GITHUB_USERNAME="$1"
GITHUB_TOKEN="$2"
MLFLOW_TRACKING_URI="$3"
MLFLOW_MODEL_NAME="$4"


REGISTRY="ghcr.io"
IMAGE_NAME="mcdoritos/laia-taxi_trip/serving"
ALIAS="production"


echo "Logging into GHCR..."
echo "$GITHUB_TOKEN" | docker login "$REGISTRY" -u "$GITHUB_USERNAME" --password-stdin

echo "Pulling production image..."
docker pull "$REGISTRY/$IMAGE_NAME:$ALIAS"


if docker ps -q -f name=serving-app >/dev/null; then
    echo "Stopping existing container..."
    docker stop serving-app || true
    docker rm serving-app || true
else
    echo "No running container found, skipping stop."
fi

echo "Starting new serving-app container..."


mkdir -p /home/admin/Desktop/Flask/serving/logs


docker run -d \
  --name serving-app \
  --restart always \
  -p 9001:8080 \
  -v /home/admin/Desktop/Flask/serving/logs:/app/logs \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e MLFLOW_MODEL_NAME="$MLFLOW_MODEL_NAME" \
  -e MODEL_ALIAS="$ALIAS" \
  "$REGISTRY/$IMAGE_NAME:$ALIAS"

echo "✅ Deployment done successfully."