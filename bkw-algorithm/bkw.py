from find_eigenvalues import diagonalize_basis
import numpy as np
from scipy.linalg import inv
import numpy as np
import sys
import os
from find_derivative import find_derivative
from find_kernel import kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import ttm,largest_modulus_coordinates_3d

def bkw_decompose(tensor):
    """
    Performs a tensor rank decomposition (minimal CP decomposition) of a third-order cubic tensor using the BKW algorithm.

    Args:
        tensor (numpy.ndarray): 
            A 3D cubic tensor of shape (n, n, n) with full multilinear rank and full rank.

    Returns:
        - factors (list of float): 
            List of diagonal elements extracted from the decomposed tensor.
        - matrices (list of numpy.ndarray): 
            List containing the three factor matrices [A, B, C].
    """
    n = len(tensor)
    derivative = np.array(find_derivative(tensor)).T 
    basis = kernel(derivative)

    permuted_factors = diagonalize_basis(basis) 
    [Ap,Bp,Cp] =  permuted_factors
    sparse = ttm(ttm(ttm(tensor,Ap.T,1),Bp.T,2),Cp.T,3) 
    non_zero_coords = largest_modulus_coordinates_3d(sparse) 
    
    permutation_tensor = np.zeros((n, n, n))
    # Place ones at the specified coordinates
    for p, q, s in non_zero_coords:
        permutation_tensor[p, q, s] = 1

    ones_vec = np.ones((1, n))
    [X1,X2] = [np.squeeze(ttm(permutation_tensor,ones_vec,i)) for i in range(1,3)] 
    factor_tensor = ttm(ttm(sparse,X2.T,1),X1.T,2)
    A = inv(Ap.T) @ X2
    B = inv(Bp.T) @ X1
    C = inv(Cp.T)
    factors =[]
    for k in range(len(tensor)):
        factors.append(factor_tensor[k,k,k])

    return factors,[A,B,C]

def bkw_recompose(factors,factor_matrices):
    """
    Performs CP recomposition of a third-order cubic tensor from given factors and factor matrices.

    Args:
        factors (list of float): 
            List of scalar factors corresponding to the diagonal entries.
        factor_matrices (list of numpy.ndarray): 
            List containing the three factor matrices [A, B, C].

    Returns:
        numpy.ndarray: 
            The reconstructed cubic tensor of shape (n, n, n).
    """
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n), dtype=complex)
    for i in range(n):
        factor_tensor[i,i,i]=factors[i]
    [A,B,C] = factor_matrices
    return ttm(ttm(ttm(factor_tensor,A,1),B,2),C,3)