from stap1 import find_derivative
from stap2 import nulruimte
from stap3 import diagonalize_basis
import numpy as np
from library import print_tensor,ttm,print_matrix,print_frontal_slices
import scipy.linalg as la

"""
tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
"""





tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
n = len(tensor)
#tensor = np.random.randn(n, n, n) # not of correct rank => generic should be rank R, using cpd
vecs = find_derivative(tensor)
matrix = np.column_stack(vecs)
basis = nulruimte(matrix)


permuted_factors = diagonalize_basis(basis) #A',B',C' 
[Ap,Bp,Cp] = [np.linalg.inv(x) for x in permuted_factors] #inverse permuted factor A'^-1,B'^-1,C'^-1
#Ap = A permuted
sparse = ttm(ttm(ttm(tensor,Ap,1),Bp,2),Cp,3)
#sparse = np.einsum('ip,jq,ks,pqs -> iqs', A, B, C, tensor)
# need to mulitply matrix in each "slicing"


def largest_modulus_coordinates_3d(tensor):
    moduli = np.abs(tensor)
    coords = []
    
    for i in range(tensor.shape[0]):
        j, k = np.unravel_index(np.argmax(moduli[i, :, :]), moduli[i, :, :].shape)
        coords.append((i, int(j), int(k)))  # Correct indexing
    return coords

#test_tensor = np.array([[[1,0],[0,0]],[[1,0],[0,0]]])

#print(largest_modulus_coordinates_3d(test_tensor))
coords = largest_modulus_coordinates_3d(sparse)

m=len(tensor) ;perm = np.zeros((m, m, m))

# Place ones at the specified coordinates
for p, q, s in coords:
    perm[p, q, s] = 1

ones_vec = np.ones((1, n))
[X1,X2,X3] = [np.squeeze(ttm(perm,ones_vec,i).T) for i in range(1,4)] #take ttm of all one vector, transpose matrix => orthogonal,
#tree matrices reordering for each direction ttm vector tensor => matrix , tensor matrix -> tensor
# noem deze X1,X2,X3

A = Ap @ X2
B = Bp @ X3
C = Cp @ X1
"""
n=2
I_tensor = np.zeros((n, n, n))
for k in range(n):
    I_tensor[k,k,k] = 1
"""

#tensor with factors on diagonal
factor_tensor = ttm(ttm(ttm(sparse,X1,1),X2,2),X3,3)

factors =[]
for k in range(len(tensor)):
    factors.append(factor_tensor[k,k,k])

X = np.zeros((n, n, n))
for i in range(n):
    t = np.einsum('i,j,k->ijk', A[:, i], B[:, i], C[:, i])
    X += tensor*(factors[i]**-1)

print_tensor(tensor)
print_tensor(X)


exit()



# print(ttm(ttm(ttm(tensor,ipf[0],1),ipf[1],2),ipf[2],3))
#null_space = la.null_space(matrix)
#print(null_space)
