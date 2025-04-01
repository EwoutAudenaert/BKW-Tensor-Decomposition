# Stap 2 van het algoritme
# We berekenen hier de kern van de afgeleide uit stap 1
import numpy as np
from scipy import linalg as la
from math import sqrt
from library import print_latex_matrix,unvectorize_tensor,print_matrix
from scipy.linalg import null_space

# Wanneer een singuliere waarde kleiner is dan 1*10^-5 keer de grootste singulier waarde, dan beschouwen we die als nul
# TODO: bekijk het nut van de drempel
def kernel(M, drempel=1e-5):
    dim = int(sqrt(len(M[0])//3))
    kernel = null_space(M)
    coefficients = np.random.randn(kernel.shape[1])
    random_vector = kernel @ coefficients
    #random_vector = kernel[:,1]    
    mat1 = random_vector[:dim**2].reshape(dim, dim)
    mat2 = random_vector[dim**2:2*(dim**2)].reshape(dim, dim)
    mat3 = random_vector[2*(dim**2):].reshape(dim, dim)
    return   [mat1,mat2,mat3]