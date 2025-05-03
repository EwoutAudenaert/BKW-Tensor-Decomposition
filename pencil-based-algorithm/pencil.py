import numpy as np
from scipy.linalg import inv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import ttm

def pencil_decompose(tensor):
    """
    Performs a pencil-based CP decomposition of a third-order tensor.

    This method uses a pencil of size 2 × n × n formed from linear combinations of slices of the tensor,
    assuming the tensor has shape (n, n, n) and full multilinear rank at least in the first two modes.

    Args:
        tensor (numpy.ndarray): 
            A 3D tensor of shape (n, n, n).

    Returns:
        list of numpy.ndarray: 
            List containing the three factor matrices [A, B, C].
    """
    n=len(tensor) 
    alpha = np.random.randn(n)
    beta = np.random.randn(n)    
    T1 = sum(alpha[i] * tensor[:, :, i] for i in range(n))
    T2 = sum(beta[i] * tensor[:, :, i] for i in range(n))

    hiddenA = T1 @ inv(T2) 
    _, A = np.linalg.eig(hiddenA)  
    hiddenBt = inv(T1) @ T2 
    _, Bt = np.linalg.eig(hiddenBt)  
    Bt = Bt.conj() 
    B  = inv(Bt.T.conj())
    Ctensor = ttm(ttm(tensor,inv(A),1),inv(B),2) 
    C = []
    for i in range(n):  
        max_j = np.argmax([np.max(np.abs(Ctensor[i, j, :])) for j in range(n)])
        C.append(Ctensor[i, max_j, :])
    #we need to transpose because np puts the row into the columns
    C = np.array(C).T
    return [A,B,C]


def pencil_recompose(factor_matrices):
    """
    Performs CP recomposition of a third-order cubic tensor from given factor matrices using the pencil method.

    Args:
        factor_matrices (list of numpy.ndarray): 
            List containing the three factor matrices [A, B, C].

    Returns:
        numpy.ndarray: 
            The reconstructed cubic tensor of shape (n, n, n).
    """
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n))
    for i in range(n):
        factor_tensor[i,i,i]=1
    [A,B,C] =factor_matrices
    return ttm(ttm(ttm(factor_tensor,A,1),B,2),C,3)

