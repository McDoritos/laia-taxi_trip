import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import glob
import pandas as pd
import pyarrow.parquet as pq

def traffic_period(hour: int) -> int:
    """
    0 = low traffic (short trips)
    1 = medium traffic
    2 = high traffic (long trips)
    """
    if 5 <= hour <= 7:
        return 0  # low
    elif (9 <= hour <= 15) or (17 <= hour <= 18):
        return 2  # high
    else:
        return 1  # medium


def read_taxi_data_for_analysis(
    root="../Dataset",
    years=("2011", "2012"),
    sample_frac_per_file=0.05,
):
    use_cols = [
        "tpep_pickup_datetime", 
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "tpep_dropoff_datetime",
        "VendorID"
    ]

    files = []
    for year in years:
        pattern = os.path.join(root, year, "**", "yellow_tripdata_*.parquet")
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError("No parquet files found")

    dfs = []
    for fpath in sorted(files):
        print(f"Reading: {fpath}")
        parquet_cols = pq.ParquetFile(fpath).schema.names
        cols = [c for c in use_cols if c in parquet_cols]

        df = pd.read_parquet(fpath, columns=cols)

        if sample_frac_per_file and 0 < sample_frac_per_file < 1:
            df = df.sample(frac=sample_frac_per_file, random_state=42)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    pickup_col = "tpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime"
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    
    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 24 * 60)]

    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dayofweek"] = df[pickup_col].dt.weekday
    df["pickup_month"] = df[pickup_col].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["traffic_period"] = df["pickup_hour"].apply(traffic_period).astype(np.int32)

    
    id_cols = ["PULocationID", "DOLocationID", "VendorID"]
    for col in id_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(np.int32)

    int_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month", 
                "is_weekend", "is_rush_hour", "passenger_count"]
    for col in int_cols:
        df[col] = df[col].fillna(0).astype(np.int32)

    float_cols = ["trip_distance"]
    for col in float_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)

    return df

df = read_taxi_data_for_analysis()

feature_cols = [
        "VendorID",
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "traffic_period",
        "PULocationID",
        "DOLocationID",
    ]

X = df[feature_cols].reset_index(drop=True)
y = df["duration_min"].values

df_corr = pd.concat([X, pd.Series(y, name="duration_min")], axis=1)
corr = df_corr.corr(numeric_only=True)

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

month_stats = (
    df.groupby("pickup_month")["duration_min"]
    .median()
)

plt.figure(figsize=(10, 4))
sns.barplot(
    x=month_stats.index,
    y=month_stats.values
)
plt.title("Median Trip Duration by Month")
plt.xlabel("Month")
plt.ylabel("Median Duration (min)")
plt.tight_layout()
plt.show()

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

dow_stats = (
    df.groupby("pickup_dayofweek")["duration_min"]
    .median()
)

plt.figure(figsize=(10, 4))
sns.barplot(
    x=[dow_labels[i] for i in dow_stats.index],
    y=dow_stats.values
)
plt.title("Median Trip Duration by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Median Duration (min)")
plt.tight_layout()
plt.show()

rush_stats = (
    df.groupby("is_rush_hour")["duration_min"]
    .median()
)

plt.figure(figsize=(5, 4))
rush_stats.plot(kind="bar")
plt.xticks([0, 1], ["Non-Rush Hour", "Rush Hour"], rotation=0)
plt.ylabel("Median Duration (min)")
plt.title("Median Trip Duration by Rush Hour")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="is_rush_hour",
    y="duration_min",
    data=df
)
plt.ylim(0, 120)
plt.xticks([0, 1], ["Non-Rush Hour", "Rush Hour"])
plt.title("Trip Duration: Rush Hour vs Non-Rush Hour")
plt.xlabel("")
plt.ylabel("Duration (minutes)")
plt.tight_layout()
plt.show()

hour_mean = df.groupby("pickup_hour")["duration_min"].mean()

plt.figure(figsize=(10, 4))
hour_mean.plot()
plt.axvspan(5, 7, alpha=0.1, label="Low traffic")
plt.axvspan(9, 15, alpha=0.1, color="red", label="High traffic")
plt.axvspan(17, 18, alpha=0.1, color="red")
plt.title("Average Trip Duration by Pickup Hour")
plt.xlabel("Hour")
plt.ylabel("Mean Duration (min)")
plt.legend()
plt.tight_layout()
plt.show()

period_map = {0: "low_duration", 1: "medium_duration", 2: "high_duration"}

mean_duration = df.groupby("traffic_period")["duration_min"].mean().rename(index=period_map)

plt.figure(figsize=(8, 5))
mean_duration.plot(kind="bar", color=["skyblue", "orange", "green"])
plt.ylim(0, 20) 
plt.ylabel("Mean Trip Duration (min)")
plt.xlabel("Rush Hour Category")
plt.title("Average Trip Duration by Traffic Period")
plt.tight_layout()
plt.show()

#Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["duration_min"], bins=100, kde=True)
plt.xlim(0, 120)
plt.title("Trip Duration Distribution (minutes)")
plt.xlabel("Duration (min)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

df["log_duration"] = np.log1p(df["duration_min"])

plt.figure(figsize=(8, 5))
sns.histplot(df["log_duration"], bins=100, kde=True)
plt.title("Log(1 + Duration) Distribution")
plt.xlabel("log(duration + 1)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="trip_distance",
    y="duration_min",
    data=df.sample(5000, random_state=42),
    alpha=0.3
)
plt.ylim(0, 120)
plt.xlim(0, 30)
plt.title("Trip Distance vs Duration")
plt.tight_layout()
plt.show()

df["distance_bin"] = pd.cut(
    df["trip_distance"],
    bins=[0, 1, 2, 5, 10, 20, 50],
)

distance_stats = (
    df.groupby("distance_bin")["duration_min"]
    .agg(["mean", "median", "count"])
)

print(distance_stats)

plt.figure(figsize=(6, 4))
sns.boxplot(x="VendorID", y="duration_min", data=df)
plt.ylim(0, 120)
plt.title("Trip Duration by Vendor")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(x="is_weekend", y="duration_min", data=df)
plt.ylim(0, 120)
plt.xticks([0, 1], ["Weekday", "Weekend"])
plt.title("Weekend vs Weekday Trip Duration")
plt.tight_layout()
plt.show()

hour_stats = (
    df.groupby("pickup_hour")["duration_min"]
    .median()
)

plt.figure(figsize=(10, 4))
hour_stats.plot(kind="bar")
plt.title("Median Trip Duration by Pickup Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Median Duration (min)")
plt.tight_layout()
plt.show()
