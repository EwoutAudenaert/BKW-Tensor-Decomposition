import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bkw-algorithm')))
from bkw import bkw_recompose, bkw_decompose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pencil-based-algorithm')))
from pencil import pencil_decompose,pencil_recompose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import largest_modulus_coordinates_2d

def random_orthogonal(n):
    """
    Generates a random orthogonal matrix of size n x n.

    Args:
        n (int): 
            Dimension of the square matrix.

    Returns:
        numpy.ndarray: 
            A random orthogonal matrix of shape (n, n).
    """
    H = np.random.randn(n, n)
    Q, _ = np.linalg.qr(H)
    return Q

def get_algo_error(algo='pencil',n=3):
    """
    Computes the forward decomposition error for a tensor decomposed with the specified algorithm.

    Args:
        algo (str, optional): 
            Algorithm to use for decomposition ('pencil' or 'bkw'). Defaults to 'pencil'.
        n (int, optional): 
            Dimension of the cubic tensor (n x n x n). Defaults to 3.

    Returns:
        float: 
            Frobenius norm of the decomposition error.
    """
    recompose = lambda a,b,c: np.einsum('i,j,k->ijk', a, b, c)
    factor_matrices = [random_orthogonal(n) for _ in range(0,3)]

    tensor=None
    re_factor_matrices=[]

    if algo =='bkw':
        tensor = bkw_recompose([i for i in range(1,n+1)],factor_matrices)
        _, re_factor_matrices = bkw_decompose(tensor)
                    
    if algo == 'pencil':
        tensor = pencil_recompose(factor_matrices)
        re_factor_matrices = pencil_decompose(tensor) 
    
    scaled_permutations = [np.linalg.solve(U, D) for U, D in zip(re_factor_matrices, factor_matrices)]
    permutations =[]
    for scaled_perm in scaled_permutations:
        coords = largest_modulus_coordinates_2d(scaled_perm)
        perm = np.zeros((n,n))
        for i, j in coords:
            perm[i, j] =  -1 if scaled_perm[i,j] <0 else 1
        permutations.append(perm)
    err =0
    re_factor_matrices = [ M @ P  for M,P in zip(re_factor_matrices,permutations)]  
    for i in range(n):
        a,b,c = [x[:,i] for x in factor_matrices]
        a_,b_,c_ = [x[:,i] for x in re_factor_matrices]
        err += np.sum((recompose(a,b,c) - recompose(a_,b_,c_)) ** 2)
    #we take the forbenius norm of the error we do this by squaring each of the terms therefor leaving out sqrt in their norm
    return np.sqrt(err)