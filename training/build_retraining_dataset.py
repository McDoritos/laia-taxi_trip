import pandas as pd

inf = pd.read_json("data/live_logs/inference_logs.jsonl", lines=True)
labels = pd.read_json("data/live_logs/labels.jsonl", lines=True)

df = inf.merge(labels, on="request_id", how="inner")

df = df.rename(columns={"true_duration": "duration_min"})

df.to_parquet("data/retraining_dataset.parquet", index=False)

print(f"Saved {len(df)} samples for retraining")