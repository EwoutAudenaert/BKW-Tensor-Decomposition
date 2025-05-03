import numpy as np
from scipy.linalg import inv, det

def diagonalize_basis(basis):
    """
    Extracts factor matrices from a basis of a vector space by diagonalizing each basis element.

    Args:
        basis (list of numpy.ndarray): 
            List of square matrices representing the basis.

    Returns:
        list of numpy.ndarray: 
            List of factor matrices obtained from the diagonalization.
    """
    results = [] 
    for X_i in basis:
        _,eigenvectors = np.linalg.eig(X_i)
        results.append(inv(eigenvectors.T)*det(eigenvectors)) 
    return results 