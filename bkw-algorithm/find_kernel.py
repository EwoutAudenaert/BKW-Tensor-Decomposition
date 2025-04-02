# Stap 2 van het algoritme
# We berekenen hier de kern van de afgeleide uit stap 1
import numpy as np
from scipy import linalg as la
from math import sqrt
from scipy.linalg import null_space

def kernel(M):
    dim = int(sqrt(len(M[0])//3))
    kernel = null_space(M)
    coefficients = np.random.randn(kernel.shape[1])
    random_vector = kernel @ coefficients
    mat1 = random_vector[:dim**2].reshape(dim, dim)
    mat2 = random_vector[dim**2:2*(dim**2)].reshape(dim, dim)
    mat3 = random_vector[2*(dim**2):].reshape(dim, dim)
    return   [mat1,mat2,mat3]