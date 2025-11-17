"""
Promote latest staging MLflow model version to production.
Intended to run after staging tests succeed.
"""

import os
from mlflow.tracking import MlflowClient

from_alias = os.getenv('FROM_ALIAS')
if not from_alias:
    raise EnvironmentError("Missing required env var: FROM_ALIAS")

to_alias = os.getenv('TO_ALIAS')
if not to_alias:
    raise EnvironmentError("Missing required env var: TO_ALIAS")

model_name = os.getenv('MLFLOW_MODEL_NAME')
if not model_name:
    raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

def promote_model():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise RuntimeError("Missing env var MLFLOW_TRACKING_URI")

    print(f"Connecting to MLflow at {mlflow_uri}")
    client = MlflowClient(tracking_uri=mlflow_uri)

    # Get model versions under the staging alias
    versions = client.get_model_version_by_alias(model_name, from_alias)
    if not versions:
        raise RuntimeError(f"No model tagged '{from_alias}' found for {model_name}")

    version = versions.version
    print(f"Found {model_name} version {version} under alias '{from_alias}'")

    # Promote
    client.set_registered_model_alias(
        name=model_name, alias=to_alias, version=version
    )

    print(f"✅ Promoted {model_name} version {version} to '{to_alias}'")


if __name__ == "__main__":
    promote_model()