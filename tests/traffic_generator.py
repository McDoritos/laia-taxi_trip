import requests
import pandas as pd
import time
import random
import json

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
            payload = pd.DataFrame([data]).to_json(orient="split")
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(URL, data=payload, headers=headers, timeout=5)
            print(f"Request {i}: {response.status_code}")
        except Exception as e:
            print(f"Failed to send request: {e}")
        
        time.sleep(0.1)

if __name__ == "__main__":
    # Send normal traffic
    send_traffic(n=40, drift=False)
    # Send some weird traffic
    send_traffic(n=10, drift=True)