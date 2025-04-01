from stap1 import find_derivative
from stap2 import kernel
from stap3 import diagonalize_basis
import numpy as np
from library import print_tensor,ttm,print_matrix,print_frontal_slices,print_latex_matrix
import scipy.linalg as la
from scipy.linalg import inv, det
import matplotlib.pyplot as plt
import numpy as np


"""
tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
"""
tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])

n = len(tensor)
derrivative = np.array(find_derivative(tensor)).T #12 by 8 matrix

basis = kernel(derrivative)
permuted_factors = diagonalize_basis(basis) #A',B',C'


[Ap,Bp,Cp] = [x for x in permuted_factors] #inverse permuted factor A'^-1,B'^-1,C'^-1

sparse = ttm(ttm(ttm(tensor,Ap.T,1),Bp.T,2),Cp.T,3)

def largest_modulus_coordinates_3d(tensor):
    moduli = np.abs(tensor)
    coords = []
    
    for i in range(tensor.shape[0]):
        j, k = np.unravel_index(np.argmax(moduli[i, :, :]), moduli[i, :, :].shape)
        coords.append((i, int(j), int(k)))  # Correct indexing
    return coords

coords = largest_modulus_coordinates_3d(sparse)


m=len(tensor) ;perm = np.zeros((m, m, m))

# Place ones at the specified coordinates
for p, q, s in coords:
    perm[p, q, s] = 1


ones_vec = np.ones((1, n))
[X1,X2,X3] = [np.squeeze(ttm(perm,ones_vec,i)) for i in range(1,4)] 
#take ttm of all one vector, transpose matrix => orthogonal,
#tree matrices reordering for each direction ttm vector tensor => matrix , tensor matrix -> tensor
# noem deze X1,X2,X3
factor_tensor = ttm(ttm(ttm(sparse,X1,1),X2,2),X3,3)
print_frontal_slices(factor_tensor)

A = (Ap @ (X1.T))
B = (Bp @ (X2.T))
C = (Cp @ (X3.T))


factors =[]
for k in range(len(tensor)):
    factors.append(factor_tensor[k,k,k])




X = ttm(ttm(ttm(factor_tensor,inv(A).T,1),inv(B).T,2),inv(C).T,3)
print_frontal_slices(X)
exit()
#X = np.einsum('az,bz,cz -> abc', A, B, C)



# print(ttm(ttm(ttm(tensor,ipf[0],1),ipf[1],2),ipf[2],3))
#null_space = la.null_space(matrix)
#print(null_space)
