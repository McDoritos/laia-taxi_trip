"""Unit tests for Flask API."""
import json
from unittest.mock import Mock, patch
import numpy as np
import requests
import os
import pytest
from serving.app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0, 1, 2]))
    return model       


def test_health_endpoint_without_model(client):
    """Test health endpoint when no model is loaded."""
    with patch.dict(app.config, {"MODEL": None}):
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is False


def test_health_endpoint_with_model(client, mock_model):
    """Test health endpoint when model is loaded."""
    with patch.dict(app.config, {"MODEL": mock_model}):
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True