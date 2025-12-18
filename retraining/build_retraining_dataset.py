import pandas as pd
import os

inf = pd.read_json("data/live_logs/inference_logs.jsonl", lines=True)
labels = pd.read_json("data/live_logs/labels.jsonl", lines=True)


inf['request_id'] = inf['request_id'].astype(str)
labels['request_id'] = labels['request_id'].astype(str)

valid_id_mask_inf = (inf['request_id'].str.len() > 0) & (inf['request_id'] != 'nan')
valid_id_mask_lbl = (labels['request_id'].str.len() > 0) & (labels['request_id'] != 'nan')

inf = inf[valid_id_mask_inf]
labels = labels[valid_id_mask_lbl]


inf['seq_id'] = inf.groupby('request_id').cumcount()

labels = labels.drop_duplicates(subset=['request_id'], keep='last')

labels_exploded = labels.explode('labels')

labels_exploded['seq_id'] = labels_exploded.groupby('request_id').cumcount()


new_data = inf.merge(labels_exploded, on=["request_id", "seq_id"], how="inner")

new_data = new_data.rename(columns={"labels": "duration_min"})

new_data = new_data.drop(columns=['seq_id'])

dataset_path = "data/retraining_dataset.parquet"


if os.path.exists(dataset_path):
    print(f"Found existing dataset at {dataset_path}. Appending...")
    existing_data = pd.read_parquet(dataset_path)
    
    combined = pd.concat([existing_data, new_data], ignore_index=True)
    
   
    combined = combined.drop_duplicates(subset=['request_id', 'duration_min'], keep='last')
else:
    print("No existing dataset found. Creating new one.")
    combined = new_data

new_data = new_data.drop(columns=['request_id'])
new_data = new_data.drop(columns=['_prediction'])


combined.to_parquet(dataset_path, index=False)

print(f"New data added: {len(new_data)}")
print(f"Total dataset size: {len(combined)}")
