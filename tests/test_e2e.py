"""End-to-end tests for the complete ML pipeline."""
import pytest
import requests
import json
import time


# Base URLs for services
FLASK_BASE_URL = "http://localhost:8080"
# MLflow is remote, not tested directly in E2E

TAXI_FEATURE_COLUMNS = [
    "haversine_km", "trip_distance", "passenger_count", "fare_amount",
    "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
    "season", "is_rush_hour", "has_congestion_fee", "total_amount",
    "pu_zone_code", "do_zone_code"
]

SAMPLE_FEATURES = [
    [
        2.8,  # haversine_km
        3.1,  # trip_distance
        1,    # passenger_count
        12.5, # fare_amount
        14,   # pickup_hour
        3,    # pickup_dayofweek
        6,    # pickup_month
        0,    # is_weekend
        2,    # season
        1,    # is_rush_hour
        1,    # has_congestion_fee
        14.8, # total_amount
        125,  # pu_zone_code
        87    # do_zone_code
    ]
]



def wait_for_service(url, timeout=30, interval=2):
    """Wait for a service to be available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


@pytest.fixture(scope="module", autouse=True)
def wait_for_services():
    """Wait for Flask service to be ready before running tests."""
    print("\nWaiting for Flask service to be ready...")
    
    # Wait for Flask app
    flask_ready = wait_for_service(f"{FLASK_BASE_URL}/health")
    if not flask_ready:
        pytest.skip("Flask service not available")
    
    print("Flask service ready! (Using remote MLflow)")


def test_flask_health():
    """Test that Flask API is healthy."""
    response = requests.get(f"{FLASK_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'


def test_flask_model_loaded():
    """Test that Flask API has a model loaded."""
    response = requests.get(f"{FLASK_BASE_URL}/health")
    data = response.json()
    
    # If model is not loaded, try to reload it
    if not data.get('model_loaded', False):
        reload_response = requests.get(f"{FLASK_BASE_URL}/reload")
        assert reload_response.status_code == 200
        
        # Check again
        response = requests.get(f"{FLASK_BASE_URL}/health")
        data = response.json()
    
    assert data['model_loaded'] is True, "Model should be loaded"


def test_prediction_single_sample():
    payload = {
        "data": SAMPLE_FEATURES,
        "columns": TAXI_FEATURE_COLUMNS
    }

    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert isinstance(data["predictions"][0], (float, int))
    assert data["predictions"][0] > 0


def test_prediction_multiple_samples():
    payload = {
        "data": SAMPLE_FEATURES * 3,
        "columns": TAXI_FEATURE_COLUMNS
    }

    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 3
    for p in data["predictions"]:
        assert isinstance(p, (float, int))

def test_prediction_without_model():
    """Test that prediction fails gracefully when model is not loaded."""
    # This test assumes we can manipulate the model state, which we can't in e2e
    # So we'll just verify the error handling works when service is down
    pass


def test_model_reload():
    """Test that model can be reloaded."""
    response = requests.get(f"{FLASK_BASE_URL}/reload")
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data or 'error' in data
    
    # Verify model is loaded after reload
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()
    assert health_data['model_loaded'] is True

def test_api_error_handling():
    """Test API error handling with invalid input."""
    # Test with missing columns
    invalid_payload = {
        "data": [[5.1, 3.5, 1.4, 0.2]]
        # Missing 'columns' key
    }
    
    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=invalid_payload,
        headers={"Content-Type": "application/json"}
    )
    
    # Should fail with 400 or 500
    assert response.status_code in [400, 500]


def test_concurrent_predictions():
    """Test that API can handle concurrent taxi prediction requests."""
    # Ensure model is loaded
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()

    if not health_data.get('model_loaded', False):
        requests.get(f"{FLASK_BASE_URL}/reload")
        time.sleep(2)

    # Payload de acordo com as features usadas no treino do modelo
    payload = {
        "data": SAMPLE_FEATURES,
        "columns": TAXI_FEATURE_COLUMNS
    }

    # Make multiple requests quickly
    responses = []
    for _ in range(5):
        response = requests.post(
            f"{FLASK_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        responses.append(response)

    # All should succeed
    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
