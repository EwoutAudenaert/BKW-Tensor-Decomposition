import numpy as np
from scipy.linalg import inv, det

def diagonalize_basis(basis, tol=1e-10):
    results = [] # time 1 ; space 1
    for X_i in basis: # time m ; space 1
        _,eigenvectors = np.linalg.eig(X_i) # time n^3 ; space n^2
        results.append(inv(eigenvectors.T)*det(eigenvectors)) # time n^3 ; space n^2
    return results # time 1 ; space m*n^2
