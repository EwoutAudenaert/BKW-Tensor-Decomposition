import numpy as np
from library  import print_matrix,tensor1,ttm,print_frontal_slices,small_tensor
from scipy.linalg import inv, det,lstsq
"""
sanity check
eigvecs_normalized = vecs / np.linalg.norm(vecs, axis=0)


W = np.array([
    [-4, 1],
    [ 3, 0]
], dtype=float)

V = np.array([
    [-0.84000779,  0.54427686],
    [-0.21043072, -0.97760877]
])
residuals = []
for i in range(W.shape[1]):
    x, _, _, _ = lstsq(V, W[:, i])
    approx = V @ x
    diff = np.linalg.norm(approx - W[:, i])
    residuals.append(diff)

print("Residuals:", residuals)
"""
tensor =  np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
#todo take random linear combination of slices
n=len(tensor)
T1,T2 = tensor[:,:,0],tensor[:,:,1]
hiddenBt = inv(T1) @ T2 #B^-T diag(_) B^T with B a factor matrix
_, Bt = np.linalg.eig(hiddenBt)  # B^T

B  = inv(Bt.T)
#
#A = T1 @ inv(Bt) # A upto scale

hiddenA = T1 @ inv(T2)
_, A = np.linalg.eig(hiddenA)  # A
A = A



Ctensor = ttm(ttm(tensor,inv(A),1),inv(B),2)

C = []
for i in range(n):
    C.append(Ctensor[i,i,:])
print(C)
#we need to transpose because np puts the row into the columns
C = np.array(C).T


def pencil_recompose(factor_matrices):
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n))
    for i in range(n):
        factor_tensor[i,i,i]=1
    [A,B,C] =[x for x in factor_matrices]
    return ttm(ttm(ttm(factor_tensor,A,1),B,2),C,3)

print_frontal_slices(pencil_recompose([A,B,C]))
