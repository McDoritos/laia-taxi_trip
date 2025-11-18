"""Unit tests for Flask API."""
import json
from unittest.mock import Mock, patch
import numpy as np
import requests
import os


API_URI = os.getenv('API_URI')
if not API_URI:
    raise EnvironmentError("Missing required env var: API_URI")

def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0, 1, 2]))
    return model


def test_health_endpoint_without_model():
    """Testa o /health quando o modelo NÃO está carregado no servidor remoto."""
    response = requests.get(f"{API_URI}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is False


def test_health_endpoint_with_model():
    """Testa o /health quando o modelo ESTÁ carregado no servidor remoto."""
    response = requests.get(f"{API_URI}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True