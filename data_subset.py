import pandas as pd
import os
import glob
import pyarrow.parquet as pq

root = "./Dataset/"
sample_frac_per_file=0.01
use_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "pickup_datetime", "dropoff_datetime",
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "trip_distance", "PULocationID", "DOLocationID",
        "passenger_count", "fare_amount", "congestion_surcharge",
        "tip_amount", "total_amount"
    ]

pattern = os.path.join(root, "**", "yellow_tripdata_*.parquet")
files = sorted(glob.glob(pattern, recursive=True))
if not files:
    raise FileNotFoundError(f"No parquet files found under {root}")

sampled_dfs = []
print("Starting to read parquet files...")
for fpath in files:
    print(f"Reading {fpath}...")
    df = pd.read_parquet(
        fpath,
        engine="pyarrow",
        columns=[c for c in use_cols if c in pq.ParquetFile(fpath).schema.names],
    )
    if sample_frac_per_file and 0 < sample_frac_per_file < 1:
        df = df.sample(frac=sample_frac_per_file, random_state=123) # 42, 123
    sampled_dfs.append(df)

df = pd.concat(sampled_dfs, ignore_index=True)

df.to_parquet("data_subset.parquet", engine="pyarrow", index=False)
