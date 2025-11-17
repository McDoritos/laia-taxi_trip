"""Unit tests for training script functionality."""
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_iris_dataset_loading():
    """Test that iris dataset can be loaded."""
    X, y = load_iris(return_X_y=True)
    
    assert X.shape[0] == 150  # Iris has 150 samples
    assert X.shape[1] == 4    # 4 features
    assert len(y) == 150
    assert set(y) == {0, 1, 2}  # 3 classes


def test_train_test_split():
    """Test that train/test split works correctly."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    assert len(X_train) == 105  # 70% of 150
    assert len(X_test) == 45    # 30% of 150
    assert len(y_train) == 105
    assert len(y_test) == 45


def test_model_training_c_0_1():
    """Test model training with C=0.1."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=200, C=0.1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    assert acc > 0.8  # Should achieve reasonable accuracy
    assert len(preds) == len(y_test)


def test_model_training_c_1_0():
    """Test model training with C=1.0."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    assert acc > 0.8
    assert len(preds) == len(y_test)


def test_model_training_c_10_0():
    """Test model training with C=10.0."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=200, C=10.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    assert acc > 0.8
    assert len(preds) == len(y_test)


def test_model_prediction_shape():
    """Test that model predictions have correct shape."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    assert preds.shape == y_test.shape
    assert all(p in [0, 1, 2] for p in preds)  # Valid class labels


def test_model_convergence():
    """Test that model converges during training."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X_train, y_train)
    
    # Check that model has been fitted (has coef_ attribute)
    assert hasattr(model, 'coef_')
    assert model.coef_.shape[0] == 3  # 3 classes
    assert model.coef_.shape[1] == 4  # 4 features


def test_best_model_selection():
    """Test logic for selecting best model based on accuracy."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = []
    for C in [0.1, 1.0, 10.0]:
        model = LogisticRegression(max_iter=200, C=C)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append({'C': C, 'accuracy': acc, 'model': model})
    
    # Find best model
    best_result = max(results, key=lambda x: x['accuracy'])
    
    assert best_result['accuracy'] > 0.8
    assert best_result['C'] in [0.1, 1.0, 10.0]
    assert isinstance(best_result['model'], LogisticRegression)


def test_feature_importance():
    """Test that trained model has reasonable feature weights."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X_train, y_train)
    
    # Check coefficients exist and are not all zero
    assert hasattr(model, 'coef_')
    assert not np.all(model.coef_ == 0)
    assert model.coef_.shape == (3, 4)  # 3 classes, 4 features

