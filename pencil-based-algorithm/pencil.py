import numpy as np
from library  import print_matrix,tensor1,ttm,print_frontal_slices
from scipy.linalg import inv, det

tensor = tensor1
#todo take random linear combination of slices
n=len(tensor)
T1,T2 = tensor[:,:,0],tensor[:,:,1]
print(det(T1))
hiddenBt = inv(T1)*T2 #B^-T diag B^T with B a factor matrix
_, Bt = np.linalg.eig(hiddenBt)  # B^T
A = T1*inv(Bt) # A upto scale
Ctensor = ttm(ttm(tensor,A,1),inv(Bt.T).T,2)
C = []
for i in range(n):
    C.append(Ctensor[i,i,:])
C = np.array(C)
print_matrix(C)
def pencil_recompose(factor_matrices):
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n))
    for i in range(n):
        factor_tensor[i,i,i]=1
    [A,B,C] = factor_matrices
    return ttm(ttm(ttm(factor_tensor,A,1),B,2),C,3)

print_frontal_slices(tensor)
print_frontal_slices(pencil_recompose([A,Bt.T,C]))
