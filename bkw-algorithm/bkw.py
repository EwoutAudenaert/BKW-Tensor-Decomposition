from find_eigenvalues import diagonalize_basis
import numpy as np
from scipy.linalg import inv
import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from find_derivative import find_derivative
from find_kernel import kernel
from library import ttm,print_frontal_slices


def largest_modulus_coordinates_3d(tensor):
    moduli = np.abs(tensor) # lineair --> O(n^3) I guess
    coords = []
    
    for i in range(tensor.shape[0]):
        j, k = np.unravel_index(np.argmax(moduli[i, :, :]), moduli[i, :, :].shape)
        coords.append((i, int(j), int(k)))  # Correct indexing
    return coords


def bkw_decompose(tensor):
    n = len(tensor)
    derivative = np.array(find_derivative(tensor)).T # Time: O(n^3)
    basis = kernel(derivative) # Time: O(n^7)

    permuted_factors = diagonalize_basis(basis) #A',B',C' # Time: O(n^3)
    [Ap,Bp,Cp] =  permuted_factors
    sparse = ttm(ttm(ttm(tensor,Ap.T,1),Bp.T,2),Cp.T,3) # Time: O(n^3)
    non_zero_coords = largest_modulus_coordinates_3d(sparse) 

    permutation_tensor = np.zeros((n, n, n))
    # Place ones at the specified coordinates
    for p, q, s in non_zero_coords:
        permutation_tensor[p, q, s] = 1

    ones_vec = np.ones((1, n))
    [X1,X2,X3] = [np.squeeze(ttm(permutation_tensor,ones_vec,i)) for i in range(1,4)] 
    factor_tensor = ttm(ttm(ttm(sparse,X1,1),X2,2),X3,3)
    A = (Ap @ (X1.T))
    B = (Bp @ (X2.T))
    C = (Cp @ (X3.T))

    factors =[]
    for k in range(len(tensor)):
        factors.append(factor_tensor[k,k,k])
    return factors,[A,B,C]
def bkw_recompose(factors,factor_matrices):
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n), dtype=complex)
    for i in range(n):
        factor_tensor[i,i,i]=factors[i]
    [A,B,C] = factor_matrices
    return ttm(ttm(ttm(factor_tensor,inv(A).T,1),inv(B).T,2),inv(C).T,3)


#tensor =  np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
#factors,matrices = bkw_decompose(tensor)
#print_frontal_slices(bkw_recompose(factors,matrices))