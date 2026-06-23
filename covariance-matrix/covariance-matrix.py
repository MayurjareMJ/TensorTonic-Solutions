import numpy as np

def covariance_matrix(X):
    X = np.asarray(X, dtype=np.float64)

    # Check if input is 2D
    if X.ndim != 2:
        return None

    N = X.shape[0]

    # Need at least 2 samples
    if N < 2:
        return None

    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Sample covariance matrix
    cov = (X_centered.T @ X_centered) / (N - 1)

    return cov