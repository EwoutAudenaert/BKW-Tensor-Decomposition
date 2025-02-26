# Stap 2 van het algoritme
# We berekenen hier de kern van de afgeleide uit stap 1
import numpy as np
from scipy import linalg as la
from math import sqrt
from library import print_latex_matrix,unvectorize_tensor,print_matrix
from scipy.linalg import null_space

# Wanneer een singuliere waarde kleiner is dan 1*10^-5 keer de grootste singulier waarde, dan beschouwen we die als nul
# TODO: bekijk het nut van de drempel
def nulruimte(M, drempel=1e-5):
    dim = int(sqrt(len(M[0])//3))
    #_, s, Vh = la.svd(M)
    #tol = drempel * max(s) if s.size > 0 else 0
    #Vh_filtered = np.where(np.abs(Vh) < tol, 0, Vh)
    kernel=np.real(null_space(M))

    coefficients = np.random.randn(kernel.shape[1])
    random_vector = kernel @ coefficients

    #hard coded for now
    mat1 = random_vector[:dim**2].reshape(dim, dim)
    mat2 = random_vector[dim**2:2*(dim**2)].reshape(dim, dim)
    mat3 = random_vector[2*(dim**2):].reshape(dim, dim)
    return   [mat1,mat2,mat3]

"""
# testcase 1: identiteit
M1 = np.eye(3)
NS1 = nulruimte(M1)

print("Test: nulruimte van de id-matrix: ")
print(NS1)
print("Vorm: ", NS1.shape)

# testcase 2: nulmatrix
M2 = np.zeros([3,3])
NS2 = nulruimte(M2)
print("Test: nulruimte van de nul-matrix: ")
print(NS2)
print("Vorm: ", NS2.shape)

"""