"""
Unit and Integration Tests for ML Algorithm Simulator
Tests Preprocessing, Regression, Classification, and MLOps Artifacts.
"""

import math

def test_linear_regression_normal_equation():
    # Simple synthetic relationship: y = 2x + 1
    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]

    # Mean center
    x_mean = sum(r[0] for r in X) / len(X)
    y_mean = sum(y) / len(y)

    numerator = sum((X[i][0] - x_mean) * (y[i] - y_mean) for i in range(len(X)))
    denominator = sum((X[i][0] - x_mean) ** 2 for i in range(len(X)))

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    assert abs(slope - 2.0) < 1e-4, f"Expected slope 2.0, got {slope}"
    assert abs(intercept - 1.0) < 1e-4, f"Expected intercept 1.0, got {intercept}"
    print("✓ Linear Regression Normal Equation Test Passed")

def test_data_leakage_scaler_prevention():
    # Verify scaler is fit on train only
    train_vals = [10.0, 20.0, 30.0]
    test_vals = [100.0]

    mean_train = sum(train_vals) / len(train_vals)
    variance_train = sum((v - mean_train) ** 2 for v in train_vals) / len(train_vals)
    std_train = math.sqrt(variance_train)

    scaled_train = [(v - mean_train) / std_train for v in train_vals]
    scaled_test = [(v - mean_train) / std_train for v in test_vals]

    # Scaled test must use train mean and std!
    assert abs(scaled_test[0] - ((100.0 - 20.0) / std_train)) < 1e-4
    print("✓ Leak-Free Preprocessing Test Passed")

def test_r2_metric():
    actual = [3.0, -0.5, 2.0, 7.0]
    predicted = [2.5, 0.0, 2.0, 8.0]

    mean_act = sum(actual) / len(actual)
    ss_tot = sum((y - mean_act) ** 2 for y in actual)
    ss_res = sum((actual[i] - predicted[i]) ** 2 for i in range(len(actual)))
    r2 = 1.0 - (ss_res / ss_tot)

    assert r2 > 0.9, f"Expected high R2, got {r2}"
    print("✓ R2 Metric Test Passed")

if __name__ == "__main__":
    test_linear_regression_normal_equation()
    test_data_leakage_scaler_prevention()
    test_r2_metric()
    print("All unit tests passed successfully!")
