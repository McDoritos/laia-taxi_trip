import requests
import pandas as pd
import time
import random
import json

# URL of your production server (Internal DNS)
# Ensure this matches the port you configured (9001 or 8080)
URL = "http://the-traffickers-internal.dei.uc.pt:9001/predict"

def get_normal_record():
    """Generates a valid, normal-looking taxi trip."""
    return {
        "vendor_id": random.choice([1, 2]),
        "pickup_datetime": "2024-01-01T12:00:00", # Example timestamp
        "passenger_count": random.randint(1, 4),
        "pickup_longitude": -73.985 + random.uniform(-0.01, 0.01),
        "pickup_latitude": 40.758 + random.uniform(-0.01, 0.01),
        "dropoff_longitude": -73.985 + random.uniform(-0.05, 0.05),
        "dropoff_latitude": 40.758 + random.uniform(-0.05, 0.05),
        "store_and_fwd_flag": "N"
    }

def get_drifted_record():
    """Generates a weird record to trigger data drift."""
    return {
        "vendor_id": 4, # Unknown vendor
        "pickup_datetime": "2024-01-01T12:00:00",
        "passenger_count": 50, # Impossible passenger count
        "pickup_longitude": 0.0, # Middle of the ocean
        "pickup_latitude": 0.0,
        "dropoff_longitude": 0.0,
        "dropoff_latitude": 0.0,
        "store_and_fwd_flag": "Y"
    }

def send_traffic(n=50, drift=False):
    print(f"Sending {n} requests. Drift Mode: {drift}")
    for i in range(n):
        data = get_drifted_record() if drift else get_normal_record()
        try:
            # We send it as a dataframe record since that's likely what your API expects
            # Adjust the wrapping based on your API's expected schema (e.g., if it expects a list)
            payload = pd.DataFrame([data]).to_json(orient="split")
            headers = {'Content-Type': 'application/json'}
            
            # Using data=payload to send raw JSON string
            response = requests.post(URL, data=payload, headers=headers, timeout=5)
            print(f"Request {i}: {response.status_code}")
        except Exception as e:
            print(f"Failed to send request: {e}")
        
        # Sleep slightly to simulate real traffic timing
        time.sleep(0.1)

if __name__ == "__main__":
    # Send mostly normal traffic
    send_traffic(n=40, drift=False)
    # Send some weird traffic to ensure your drift report isn't empty
    send_traffic(n=10, drift=True)