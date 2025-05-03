import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm,vectorize_tensor

def unit_matrix(i, j, shape):
    """
    Creates a matrix of given shape with a single one at position (i, j) and zeros elsewhere.

    Args:
        i (int): 
            Row index where the one is placed.
        j (int): 
            Column index where the one is placed.
        shape (int): 
            Size of the square matrix (shape x shape).

    Returns:
        numpy.ndarray: 
            The resulting matrix with a single one and zeros elsewhere.
    """
    M = np.zeros((shape,shape))
    M[i, j] = 1 
    return M 

def find_derivative(tensor):
    """
    Finds the derivative of the mapping induced by a third-order tensor on triplets of matrices.

    Args:
        tensor (numpy.ndarray): 
            A 3D cubic tensor of shape (n, n, n).

    Returns:
        list of numpy.ndarray: 
            List of vectorized derivatives corresponding to each basis matrix.
    """
    dim = tensor.shape[0] 
    matrix=[] 
    for k in range(1,4):
        for i in range(0,dim): 
            for j in range(0,dim):
                u = unit_matrix(i,j,dim) 
                matrix.append(vectorize_tensor(ttm(tensor,u,k)).T) 
    return matrix