import numpy as np
from math import sqrt
from scipy.linalg import null_space

def kernel(M):
    """
    Finds matrices that together form a reshaped version of an element in the kernel of a tensor derivative matrix.

    Args:
        M (numpy.ndarray): 
            Matrix representing a tensor derivative.

    Returns:
        list of numpy.ndarray: 
            List containing three matrices reconstructed from a random element of the kernel.
    """
    dim = int(sqrt(len(M[0])//3))
    kernel = null_space(M) 
    coefficients = np.random.randn(kernel.shape[1]) 
    random_vector = kernel @ coefficients 
    mat1 = random_vector[:dim**2].reshape(dim, dim) 
    mat2 = random_vector[dim**2:2*(dim**2)].reshape(dim, dim) 
    mat3 = random_vector[2*(dim**2):].reshape(dim, dim) 
    return   [mat1,mat2,mat3] 