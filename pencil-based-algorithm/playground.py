import numpy as np
from numpy.linalg import inv,det

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import print_matrix,print_frontal_slices,largest_modulus_coordinates_2d
from pencil import pencil_decompose,pencil_recompose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bkw-algorithm')))
from bkw import bkw_recompose, bkw_decompose
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def nearly_identical_orthogonal_tensor(n,eps=1e-6):
    Q1, _ = np.linalg.qr(np.random.randn(n, n))
    Q2, _ = np.linalg.qr(np.random.randn(n, n))    
    T1 = Q1 @ Q2.T
    # T2 should be almost T1, but we keep it ortho
    perturb = np.eye(n) + eps * np.random.randn(n, n)
    U, _ = np.linalg.qr(perturb)
    T2 = T1 @ U 
    tensor = np.zeros((n, n, n))
    tensor[:, :, 0] = T1
    tensor[:, :, 1] = T2
    for k in range(2, n):
        tensor[:, :, k] = np.eye(n)  
    return tensor




def random_orthogonal(n):
    H = np.random.randn(n, n)
    Q, _ = np.linalg.qr(H)
    return Q
def close_eigs_matrices(n, a,start,end):
    Q1 = random_orthogonal(n)
    Q2 = random_orthogonal(n)
    eigvals1 = np.linspace(start, end, n)
    eigvals2 = eigvals1 + a 
    D1 = np.diag(eigvals1)
    D2 = np.diag(eigvals2)
    A1 = Q1 @ D1 @ Q1.T
    A2 = Q1 @ D2 @ Q1.T
    return A1, A2

def close_eigs_tensor(n,a,start=0.1,end=1):
    T1,T2 = close_eigs_matrices(n,a,start,end)
    tensor = np.zeros((n, n, n))
    tensor[:, :, 0] = T1
    tensor[:, :, 1] = T2
    for k in range(2, n):
        tensor[:, :, k] = random_orthogonal(n)
    return tensor

reps=1000
errs = []
start =3
end=10
recompose = lambda a,b,c: np.einsum('i,j,k->ijk', a, b, c)

for n in range(start,end):
    acc_err = 0
    for rep in range(0,reps):
        factor_matrices = [random_orthogonal(n) for _ in range(0,3)]

        #tensor = bkw_recompose([1 + i*(10**-12) for i in range(n)],factor_matrices) remove transpose if this needs to work
        #_, re_factor_matrices = bkw_decompose(tensor)
        tensor = pencil_recompose(factor_matrices)
        re_factor_matrices = pencil_decompose(tensor) 

        scaled_permutations = [np.linalg.solve(U, D) for U, D in zip(re_factor_matrices, factor_matrices)]
        permutations =[]
        for scaled_perm in scaled_permutations:
            coords = largest_modulus_coordinates_2d(scaled_perm)
            perm = np.zeros((n,n))
            for i, j in coords:
                perm[i, j] = scaled_perm[i,j]
            permutations.append(perm.T)
        #permutations = scaled_permutations


        err =0
        re_factor_matrices = [ M @ P  for M,P in zip(re_factor_matrices,permutations)]  
        for i in range(n):
            a,b,c = [x[:,i] for x in factor_matrices]
            a_,b_,c_ = [x[:,i] for x in re_factor_matrices]
            #print("=========================")
            #print_frontal_slices(abs(recompose(a,b,c)))
            #print_frontal_slices(abs(recompose(a_,b_,c_)))
        
            err += np.sqrt(np.sum(np.abs(recompose(a,b,c) - recompose(a_,b_,c_)) ** 2))
        #print(err)
        acc_err+=err

        #print_frontal_slices(tensor-pencil_recompose(factor_matrices))
        #acc_err += np.sqrt(np.sum((tensor - bkw_recompose([1 for i in range(n)],factor_matrices))**2))
    errs.append(acc_err/reps)

dims = np.arange(start, end)
r = errs 
sns.set(style="whitegrid")
sns.lineplot(x=dims, y=r, marker="o", label="Observed")
plt.xlabel("Dimensions")
plt.ylabel("Error")
plt.title("Decomposition Error vs Dimensions")
plt.legend()
plt.show()
