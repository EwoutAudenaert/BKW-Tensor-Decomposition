
import numpy as np
from scipy.linalg import inv, det

def diagonalize_basis(basis, tol=1e-10):
    results = []
    for X_i in basis:
        _,eigenvectors = np.linalg.eig(X_i)  
        results.append(inv(eigenvectors.T)*det(eigenvectors))
    return results