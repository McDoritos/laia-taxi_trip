import pandas as pd
import os

# 1. Load Data
inf = pd.read_json("data/live_logs/inference_logs.jsonl", lines=True)
labels = pd.read_json("data/live_logs/labels.jsonl", lines=True)

# 2. SANITIZATION STEP
inf['request_id'] = inf['request_id'].astype(str)
labels['request_id'] = labels['request_id'].astype(str)

valid_id_mask_inf = (inf['request_id'].str.len() > 0) & (inf['request_id'] != 'nan')
valid_id_mask_lbl = (labels['request_id'].str.len() > 0) & (labels['request_id'] != 'nan')

inf = inf[valid_id_mask_inf]
labels = labels[valid_id_mask_lbl]

# 3. PREPARE FOR ALIGNMENT
# A. Inference: Do NOT drop duplicates by request_id alone, as that deletes the sequence.
# Instead, we assume the rows are in order and assign a sequence number (0, 1, 2...)
# If you suspect exact duplicate logs (retries of the exact same step), use drop_duplicates() without subset first.
# inf = inf.drop_duplicates() 
inf['seq_id'] = inf.groupby('request_id').cumcount()

# B. Labels: Deduplicate unique request_ids (keep last), then EXPLODE the list
labels = labels.drop_duplicates(subset=['request_id'], keep='last')

# Assuming the column with the list is named 'labels' based on your later rename code. 
# If it is named 'labels', change 'labels' to 'labels' below.
labels_exploded = labels.explode('labels')

# Assign sequence number to the exploded labels so they match the inference rows
labels_exploded['seq_id'] = labels_exploded.groupby('request_id').cumcount()

# 4. Safe Merge (Match on Request ID AND Sequence ID)
# This ensures 1st log gets 1st label, 2nd log gets 2nd label.
new_data = inf.merge(labels_exploded, on=["request_id", "seq_id"], how="inner")

new_data = new_data.rename(columns={"true_duration": "duration_min"})

# Clean up helper column
new_data = new_data.drop(columns=['seq_id'])

dataset_path = "data/retraining_dataset.parquet"

# 5. Load "Historical" data (if exists) and Append
if os.path.exists(dataset_path):
    print(f"Found existing dataset at {dataset_path}. Appending...")
    existing_data = pd.read_parquet(dataset_path)
    
    combined = pd.concat([existing_data, new_data], ignore_index=True)
    
    # 6. Deduplication Strategy
    # Now that data is flattened, we can't just deduplicate by 'request_id'.
    # We must deduplicate by 'request_id' AND the implicit sequence (or just all columns).
    combined = combined.drop_duplicates(subset=['request_id', 'duration_min'], keep='last')
else:
    print("No existing dataset found. Creating new one.")
    combined = new_data

# 7. Save
combined.to_parquet(dataset_path, index=False)

print(f"New data added: {len(new_data)}")
print(f"Total dataset size: {len(combined)}")