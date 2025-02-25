from stap1 import find_derivative
from stap2 import nulruimte
from stap3 import diagonalize_basis
import numpy as np
from library import print_tensor,ttm,print_matrix
import scipy.linalg as la

"""
tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
"""



n=4
tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
#tensor = np.random.randn(n, n, n)

vecs = find_derivative(tensor)
matrix = np.column_stack(vecs)
#print_latex_matrix_int(matrix)

basis = nulruimte(matrix)

#for b in basis:
#    print_latex_matrix(b)
permuted_factors = diagonalize_basis(basis) #A',B',C'
[A,B,C] = [np.linalg.inv(x) for x in permuted_factors] #inverse permuted factor A'^-1,B'^-1,C'^-1

image = np.einsum('ip,jq,kr,pqs -> iqs', A, B, C, tensor)
print_tensor(image)
exit()

def largest_modulus_coordinates_3d(tensor):
    moduli = np.abs(tensor)
    coords = []
    
    for i in range(tensor.shape[0]):
        k, j = np.unravel_index(np.argmax(moduli[i, :, :]), moduli[i, :, :].shape)
        coords.append(((i, int(j), int(k)),tensor[i][j][k]))
    return coords

coords = largest_modulus_coordinates_3d(image)


m=len(tensor) ;perm = np.zeros((m, m, m))

# Place ones at the specified coordinates
for p, q, s in coords:
    tensor[p, q, s] = 1


# print(ttm(ttm(ttm(tensor,ipf[0],1),ipf[1],2),ipf[2],3))
#null_space = la.null_space(matrix)
#print(null_space)
