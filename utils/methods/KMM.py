import numpy as np
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

kernel_type='rbf'
bandwidth=1

def iwe_kernel_mean_matching(X, Z):
    """
    Estimate importance weights based on kernel mean matching.
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Returns
    -------
    iw : array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Compute sample pairwise distances
    KXX = cdist(X, X, metric='euclidean')
    KXZ = cdist(X, Z, metric='euclidean')

    # Check non-negative distances
    if not np.all(KXX >= 0):
        raise ValueError('Non-positive distance in source kernel.')
    if not np.all(KXZ >= 0):
        raise ValueError('Non-positive distance in source-target kernel.')

    # Compute kernels
    if kernel_type == 'rbf':
        # Radial basis functions
        KXX = np.exp(-KXX / (2 * bandwidth ** 2))
        KXZ = np.exp(-KXZ / (2 * bandwidth ** 2))

    # Collapse second kernel and normalize
    KXZ = N / M * np.sum(KXZ, axis=1)

    # Prepare for CVXOPT
    Q = matrix(KXX, tc='d')
    p = matrix(KXZ, tc='d')
    G = matrix(np.concatenate((np.ones((1, N)), -1 * np.ones((1, N)),
                               -1. * np.eye(N)), axis=0), tc='d')
    h = matrix(np.concatenate((np.array([N / np.sqrt(N) + N], ndmin=2),
                               np.array([N / np.sqrt(N) - N], ndmin=2),
                               np.zeros((N, 1))), axis=0), tc='d')

    # Call quadratic program solver
    sol = solvers.qp(Q, p, G, h)

    # Return optimal coefficients as importance weights
    return np.array(sol['x'])[:, 0]