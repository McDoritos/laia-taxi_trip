# This script simulates the entire lifecycle for the Defense
import subprocess

def test_full_drift_lifecycle():
    print("=== STEP 1: TRAINING INITIAL MODEL (JAN-OCT) ===")
    subprocess.run(["python", "train_model.py", "--simulate-old-data"])
    
    print("\n=== STEP 2: SIMULATING NEW TRAFFIC (NOV) ===")

    subprocess.run(["python", "tests/traffic_generator.py", "--drift-mode"])
    
    print("\n=== STEP 3: RUNNING MONITORING ===")

    result = subprocess.run(["python", "monitoring/check_drift.py"], capture_output=True, text=True)
    print(result.stdout)
    
    assert "DATA DRIFT DETECTED" in result.stdout
    assert "Triggering retraining" in result.stdout
    
    print("\n=== SUCCESS: Pipeline correctly detected drift and triggered retraining! ===")

if __name__ == "__main__":
    test_full_drift_lifecycle()