import numpy as np


def normalize_train(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    X_norm = (X - min_vals) / ranges

    return X_norm, min_vals, max_vals


def normalize_test(X, min_vals, max_vals):
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    return (X - min_vals) / ranges