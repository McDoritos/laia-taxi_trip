"""Unit tests for Flask API."""
import pytest
import json
from unittest.mock import Mock, patch
import numpy as np
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


def test_predict_without_model(client):
    """Test predict endpoint when no model is loaded."""
    with patch.dict(app.config, {"MODEL": None}):
        response = client.post(
            '/predict',
            data=json.dumps({
                'data': [[5.1, 3.5, 1.4, 0.2]],
                'columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            }),
            content_type='application/json'
        )
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Model not loaded' in data['error']


def test_predict_with_model(client, mock_model):
    """Test predict endpoint with a loaded model."""
    with patch.dict(app.config, {"MODEL": mock_model}):
        response = client.post(
            '/predict',
            data=json.dumps({
                'data': [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.3, 3.0, 6.3, 1.8]],
                'columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data
        assert len(data['predictions']) == 3
        assert data['predictions'] == [0, 1, 2]


def test_predict_invalid_data(client, mock_model):
    """Test predict endpoint with invalid data format."""
    with patch.dict(app.config, {"MODEL": mock_model}):
        response = client.post(
            '/predict',
            data=json.dumps({'invalid': 'data'}),
            content_type='application/json'
        )
        # This should fail with 400 or 500 depending on error handling
        assert response.status_code in [400, 500]


def test_reload_model_success(client):
    """Test reload endpoint when model loads successfully."""
    with patch('mlflow.pyfunc.load_model') as mock_load:
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        response = client.get('/reload')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'message' in data
        assert 'reloaded successfully' in data['message']


def test_reload_model_failure(client):
    """Test reload endpoint when model fails to load."""
    with patch('mlflow.pyfunc.load_model') as mock_load:
        mock_load.side_effect = Exception("MLflow error")
        
        response = client.get('/reload')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to load model' in data['error']


def test_predict_correct_dataframe_creation(client, mock_model):
    """Test that predict correctly creates DataFrame from request."""
    with patch.dict(app.config, {"MODEL": mock_model}):
        test_data = [[5.1, 3.5, 1.4, 0.2]]
        test_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        response = client.post(
            '/predict',
            data=json.dumps({
                'data': test_data,
                'columns': test_columns
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        # Verify that model.predict was called
        mock_model.predict.assert_called_once()

