import pandas as pd
import os

import pandas as pd
import numpy as np

# 1. Load Data
inf = pd.read_json("data/live_logs/inference_logs.jsonl", lines=True)
labels = pd.read_json("data/live_logs/labels.jsonl", lines=True)

# 2. SANITIZATION STEP (The missing guarantee)
# Convert to string to ensure type safety
inf['request_id'] = inf['request_id'].astype(str)
labels['request_id'] = labels['request_id'].astype(str)

# Filter out empty strings, None, or "nan" strings
valid_id_mask_inf = (inf['request_id'].str.len() > 0) & (inf['request_id'] != 'nan')
valid_id_mask_lbl = (labels['request_id'].str.len() > 0) & (labels['request_id'] != 'nan')

inf = inf[valid_id_mask_inf]
labels = labels[valid_id_mask_lbl]

# 3. Deduplicate (Prevent Cartesian explosion)
# If duplicates exist, keep the last one (assuming it's the most recent retry)
inf = inf.drop_duplicates(subset=['request_id'], keep='last')
labels = labels.drop_duplicates(subset=['request_id'], keep='last')

# 4. Safe Merge
new_data = inf.merge(labels, on="request_id", how="inner")
new_data = new_data.rename(columns={"true_duration": "duration_min"})

dataset_path = "data/retraining_dataset.parquet"

# 2. Load "Historical" data (if exists) and Append
if os.path.exists(dataset_path):
    print(f"Found existing dataset at {dataset_path}. Appending...")
    existing_data = pd.read_parquet(dataset_path)
    
    # Concatenate old + new
    combined = pd.concat([existing_data, new_data], ignore_index=True)
    
    # 3. Deduplicate (Safety net: prevents duplicates if workflow runs twice)
    # We keep 'last' to ensure we have the most updated version of a record
    combined = combined.drop_duplicates(subset=['request_id'], keep='last')
else:
    print("No existing dataset found. Creating new one.")
    combined = new_data

# 4. Save
combined.to_parquet(dataset_path, index=False)

print(f"New data added: {len(new_data)}")
print(f"Total dataset size: {len(combined)}")