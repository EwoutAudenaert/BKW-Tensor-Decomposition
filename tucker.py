import numpy as np
from library import ttm
def unfold(X, mode) :
    size_X = X.shape
    n_dims = X.ndim

    if mode < 0 or mode >= n_dims:
        raise ValueError(f"Mode must be between 0 and {n_dims - 1}")

    perm = [mode] + [i for i in range(n_dims) if i != mode]

    X_perm = np.transpose(X, perm)
    rows = size_X[mode]
    cols = np.prod(size_X) // rows
    return X_perm.reshape(rows, cols)

def mlp(X, factors):
    for mode, U in enumerate(factors):
        X = np.tensordot(U, X, axes=(1, mode))
    return X

def tucker(X,threshold=1e-10):
    n_dims = X.ndim
    factors = [None] * n_dims

    for n in range(n_dims):
        X_n = unfold(X, n)
        U, S, _ = np.linalg.svd(X_n)

        i = 0
        try:
            threshold=1e-10
            while S[i] > threshold:
                i += 1
        except IndexError:
            pass

        factors[n] = U[:, :i].T 
    core = mlp(X, factors)
    return core, factors


