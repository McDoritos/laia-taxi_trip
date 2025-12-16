import pytest
import requests
import json
import time
import os

FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://localhost:9001")

def wait_for_service(url, timeout=30, interval=2):
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
    print("\nWaiting for Flask service to be ready...")

    ready = wait_for_service(f"{FLASK_BASE_URL}/health", timeout=40, interval=2)
    assert ready, "Flask service not available after waiting"

    print("Flask service ready!")

SAMPLE_FEATURES = [
    {
            "VendorID": 2,
            "tpep_pickup_datetime": "2011-01-01 00:10:00",
            "passenger_count": 4,
            "trip_distance": 1.2,
            "PULocationID": 145,
            "DOLocationID": 145
    }
]


def test_flask_health():
    response = requests.get(f"{FLASK_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'


def test_flask_model_loaded():
    response = requests.get(f"{FLASK_BASE_URL}/health")
    data = response.json()
    
    if not data.get('model_loaded', False):
        reload_response = requests.get(f"{FLASK_BASE_URL}/reload")
        assert reload_response.status_code == 200
        
        response = requests.get(f"{FLASK_BASE_URL}/health")
        data = response.json()
    
    assert data['model_loaded'] is True, "Model should be loaded"


def test_prediction_single_sample():
    payload = {
        "data": SAMPLE_FEATURES
    }

    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()
    preds = data if isinstance(data, list) else data.get("predictions", [])

    assert len(preds) == 1
    assert isinstance(preds[0], (float, int))
    assert preds[0] > 0


def test_prediction_multiple_samples():
    payload = {
        "data": SAMPLE_FEATURES * 3
    }

    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()
    preds = data if isinstance(data, list) else data.get("predictions", [])

    assert len(preds) == 3
    for p in preds:
        assert isinstance(p, (float, int))

def test_prediction_without_model():
    pass


def test_model_reload():
    response = requests.get(f"{FLASK_BASE_URL}/reload")
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data or 'error' in data

    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()
    assert health_data['model_loaded'] is True

def test_api_error_handling():
    invalid_payload = {
        "data": [[5.1, 3.5, 1.4, 0.2]]
    }
    
    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=invalid_payload,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code in [400, 500]


def test_concurrent_predictions():
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()

    if not health_data.get('model_loaded', False):
        requests.get(f"{FLASK_BASE_URL}/reload")
        time.sleep(2)

    payload = {
        "data": SAMPLE_FEATURES
    }

    responses = []
    for _ in range(5):
        response = requests.post(
            f"{FLASK_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        responses.append(response)

    for response in responses:
        assert response.status_code == 200
        data = response.json()

        preds = data if isinstance(data, list) else data.get("predictions", [])

        assert isinstance(preds, list)
        assert all(isinstance(p, (float, int)) for p in preds)
        assert len(preds) == 1 
