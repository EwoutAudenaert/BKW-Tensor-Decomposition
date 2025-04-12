import numpy as np
from library  import print_matrix,tensor1,ttm,print_frontal_slices,small_tensor
from scipy.linalg import inv, det,lstsq
"""

f1 = [  [1,0,0],
        [0,5,6],
        [0,8,9]]

f2 = [  [1,0,0],
        [0,4,5],
        [0,6,8]]
f3 =[  [0,0,1],
        [0,1,4],
        [1,0,0]]
        tensor = np.stack([f1, f2, f3], axis=2)
"""
f1 = [[1, -2],
      [2,1]]
f2 = [[1,-1],
      [1, 1]]

tensor =np.stack([f1, f2], axis=2, dtype=complex)
print_frontal_slices(tensor)

#todo take random linear combination of slices
n=len(tensor)
T1,T2 = tensor[:,:,0],tensor[:,:,1]

hiddenA = T1 @ inv(T2)
eigsA, A = np.linalg.eig(hiddenA)  # A
hiddenBt = inv(T1) @ T2 #B^-T diag(_) B^T with B a factor matrix
eigsB, Bt = np.linalg.eig(hiddenBt)  # B^T
Bt = Bt.conj()
B  = inv(Bt.T)

Ctensor = ttm(ttm(tensor,inv(A),1),inv(B),2)

C = []
for i in range(n):
    C.append(Ctensor[i,i,:])
    

#we need to transpose because np puts the row into the columns
C = np.array(C).T

def pencil_recompose(factor_matrices):
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n))
    for i in range(n):
        factor_tensor[i,i,i]=1
    [A,B,C] =factor_matrices
    return ttm(ttm(ttm(factor_tensor,A,1),B,2),C,3)

print_frontal_slices(pencil_recompose([A,B,C]))
