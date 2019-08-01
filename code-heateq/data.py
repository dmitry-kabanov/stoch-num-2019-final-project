"""Data generation."""
import numpy as np


def get_data(N, with_noise=False):
    """
    Generate data and return a subset with `N` elements in it.

    Parameters
    ----------
    N : int
        Number of observations to return from the generated data.
    with_noise : bool (optional, default False)
        If True, then add Gaussian noise to the generated data.

    Returns
    -------
    X_observed : ndarray (N, 2)
        Matrix of the features.
    u_observed : ndarray (N,)
        Target vector.
    lb, ub : tuple    
        Tuple of minima and maxima in (x, t).
    
    """
    X_star, u_star, __, __, __, __, __ = get_all_data(with_noise)

    # Doman bounds
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)

    if N > 0.9*X_star.shape[0]:
        raise ValueError('Number of observations is too big')

    idx = np.random.choice(X_star.shape[0], N, replace=False)
    X_observed = X_star[idx, :]
    u_observed = u_star[idx, :]

    return X_observed, u_observed, lb, ub


def get_all_data(with_noise=False):
    """
    Generate data and return all of them.

    Parameters
    ----------
    with_noise : bool (optional, default False)
        If True, then add Gaussian noise to the generated data.

    Returns
    -------
    X_star : ndarray (N, 2)
        Matrix of the features.
    u_star : ndarray (N,)
        Target vector.
    
    """
    x = np.linspace(-1, 1, num=201)
    t = np.linspace(0, 1.0, num=101)

    X, T = np.meshgrid(x, t)
    Exact = np.exp(-T) * np.sin(2*np.pi*X)
    # Exact = np.exp(-X**2 / (1 + 4*T)) / np.sqrt(1 + 4*T)
    # Exact = np.exp(-T) * np.exp(-X**2)

    if with_noise:
        Exact = Exact*(1 + 0.05*np.random.normal(size=Exact.shape))

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    return X_star, u_star, x, t, Exact, X, T
