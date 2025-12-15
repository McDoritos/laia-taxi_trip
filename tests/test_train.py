"""Unit tests for taxi trip training pipeline."""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

@pytest.fixture(scope="module")
def small_dataset():
    path = "Dataset/data_subset.parquet"
    df = pd.read_parquet(path)

    if "duration_min" not in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        df["duration_min"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
            .dt.total_seconds() / 60
        )
        df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 24 * 60)]

    if "pickup_hour" not in df.columns:
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    if "pickup_dayofweek" not in df.columns:
        df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.weekday
    if "pickup_month" not in df.columns:
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    if "is_rush_hour" not in df.columns:
        df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    def traffic_period(hour: int) -> int:
        """0 = low, 1 = medium, 2 = high traffic intensity"""
        if 5 <= hour <= 7:
            return 0
        elif (9 <= hour <= 15) or (17 <= hour <= 18):
            return 2
        else:
            return 1

    df["traffic_period"] = df["pickup_hour"].apply(traffic_period).astype(np.int32)

    feature_cols = [
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

    X = df[feature_cols].fillna(0).reset_index(drop=True)
    y = df["duration_min"].values
    return X, y


def test_dataset_columns(small_dataset):
    X, y = small_dataset
    expected_cols = [
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
    for col in expected_cols:
        assert col in X.columns, f"Missing feature: {col}"
    assert len(y) == len(X), "Target length mismatch"


def test_feature_sanity(small_dataset):
    X, _ = small_dataset
    assert X["pickup_hour"].between(0, 23).all()
    assert X["pickup_dayofweek"].between(0, 6).all()
    assert set(X["is_weekend"].unique()).issubset({0, 1})
    assert set(X["is_rush_hour"].unique()).issubset({0, 1})
    assert X["traffic_period"].between(0, 2).all()
    assert X["trip_distance"].min() >= 0
    assert X["passenger_count"].min() >= 0


def test_lgbm_training_smoke(small_dataset):
    X, y = small_dataset
    model = LGBMRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all(preds > 0)
    mae = mean_absolute_error(y, preds)
    assert isinstance(mae, float)


def test_random_forest_training_smoke(small_dataset):
    X, y = small_dataset
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all(preds > 0)
    mae = mean_absolute_error(y, preds)
    assert isinstance(mae, float)


def test_ab_model_selection(small_dataset):
    X, y = small_dataset
    model_a = LGBMRegressor(n_estimators=5, random_state=1)
    model_b = LGBMRegressor(n_estimators=5, random_state=2)
    model_a.fit(X, y)
    model_b.fit(X, y)

    preds_a = model_a.predict(X)
    preds_b = model_b.predict(X)
    mae_a = mean_absolute_error(y, preds_a)
    mae_b = mean_absolute_error(y, preds_b)

    assert isinstance(mae_a, float)
    assert isinstance(mae_b, float)
    assert mae_a >= 0
    assert mae_b >= 0
