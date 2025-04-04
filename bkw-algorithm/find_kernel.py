# Stap 2 van het algoritme
# We berekenen hier de kern van de afgeleide uit stap 1
import numpy as np
from scipy import linalg as la
from math import sqrt
from scipy.linalg import null_space

def kernel(M): #time r * n^4 ; space n^4
    dim = int(sqrt(len(M[0])//3)) # time 1 ; space 1
    kernel = null_space(M) # time r * n^4 ; space n^4 (worst) --> domineert
    coefficients = np.random.randn(kernel.shape[1]) # time n^2 ; space n^2
    random_vector = kernel @ coefficients # time n^4 ; space n^2
    mat1 = random_vector[:dim**2].reshape(dim, dim) # time 1 ; space n^2
    mat2 = random_vector[dim**2:2*(dim**2)].reshape(dim, dim) # time 1 ; space n^2
    mat3 = random_vector[2*(dim**2):].reshape(dim, dim) # time 1 ; space n^2
    return   [mat1,mat2,mat3] # time 1 ; space n^2
