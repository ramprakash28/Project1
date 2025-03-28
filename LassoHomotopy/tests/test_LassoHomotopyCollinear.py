import csv
import numpy as np # type: ignore
import pytest # type: ignore


from model.LassoHomotopy import LassoHomotopyModel

def test_lasso_sparsity_on_collinear_data():
    """
    Tests if LASSO produces a sparse solution on collinear data.
    """
    dataset = "LassoHomotopy/tests/collinear_data.csv"

    # Load dataset
    data = []
    with open(dataset, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if all(v.strip() != "" for v in row.values()):  # skip incomplete rows
                data.append({k: float(v) for k, v in row.items()})

    # Extract X and y
    X = np.array([[datum[k] for k in datum if k.startswith('X')] for datum in data])
    y = np.array([datum['target'] for datum in data])

    # Train LASSO with 5-Fold Cross-Validation
    model = LassoHomotopyModel()
    results = model.fit(X, y)

    # Predictions
    preds = results.predict(X)

    # Assertions
    assert preds.shape == y.shape, "Prediction shape should match y"
    assert np.any(results.coef_ != 0), "At least one coefficient should be nonzero"
    assert model.best_lambda_ is not None, "Best lambda should be selected"

    # Ensure sparsity (some coefficients should be zero)
    zero_coefs = np.sum(np.abs(results.coef_) < 1e-6)
    assert zero_coefs > 0, "LASSO should produce a sparse solution (some coefficients should be zero)"
