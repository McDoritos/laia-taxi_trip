"""Unit tests for taxi trip training pipeline."""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from your_module import read_dataset  # <-- replace with your actual import path


@pytest.fixture(scope="module")
def small_dataset():
    """Load a small sample dataset for testing."""
    X, y = read_dataset(sample_frac_per_file=0.01)
    return X, y


def test_dataset_columns(small_dataset):
    """Test that dataset has all required features."""
    X, y = small_dataset
    expected_cols = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "PULocationID",
        "DOLocationID",
    ]
    for col in expected_cols:
        assert col in X.columns, f"Missing feature: {col}"
    assert len(y) == len(X), "Target length mismatch"


def test_feature_sanity(small_dataset):
    """Test that derived features have valid ranges."""
    X, _ = small_dataset
    assert X["pickup_hour"].between(0, 23).all()
    assert X["pickup_dayofweek"].between(0, 6).all()
    assert set(X["is_weekend"].unique()).issubset({0, 1})
    assert set(X["is_rush_hour"].unique()).issubset({0, 1})
    assert X["trip_distance"].min() >= 0
    assert X["passenger_count"].min() >= 0


def test_lgbm_training_smoke(small_dataset):
    """Smoke test LightGBM training."""
    X, y = small_dataset
    model = LGBMRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    # Predictions should match input length and be positive
    assert preds.shape[0] == X.shape[0]
    assert np.all(preds > 0)
    # Check that MAE can be computed
    mae = mean_absolute_error(y, preds)
    assert isinstance(mae, float)


def test_random_forest_training_smoke(small_dataset):
    """Smoke test RandomForest training."""
    X, y = small_dataset
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all(preds > 0)
    mae = mean_absolute_error(y, preds)
    assert isinstance(mae, float)


def test_ab_model_selection(small_dataset):
    """Simulate A/B test logic between two models."""
    X, y = small_dataset
    model_a = LGBMRegressor(n_estimators=5, random_state=1)
    model_b = LGBMRegressor(n_estimators=5, random_state=2)
    model_a.fit(X, y)
    model_b.fit(X, y)

    preds_a = model_a.predict(X)
    preds_b = model_b.predict(X)
    mae_a = mean_absolute_error(y, preds_a)
    mae_b = mean_absolute_error(y, preds_b)

    # Ensure metrics are valid numbers
    assert isinstance(mae_a, float)
    assert isinstance(mae_b, float)
    # One model should be better (or tie)
    assert mae_a >= 0
    assert mae_b >= 0
